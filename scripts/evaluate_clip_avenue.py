"""Evaluate five fixed CLIP prompts across the 18 regular Avenue test videos.

This exploratory diagnostic is not the AnyAnomaly Qwen/vLLM baseline.
Run from the project root: python -m scripts.evaluate_clip_avenue
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts.clip_single_video_diagnostic import (
    CLIP_LENGTH,
    EVENTS,
    FRAMES_ROOT,
    LABEL_ROOT,
    MODEL_PATH,
    PROJECT_ROOT,
    load_labels,
    numbered_frames,
    summarize_scores,
)

PROMPTS_FILE = PROJECT_ROOT / "configs/avenue_clip_prompts.json"
VIDEO_GROUPS_FILE = PROJECT_ROOT / "third_party/Paper-AnyAnomaly/ground_truth/c-avenue.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs/clip_avenue_5events.json"


def aggregate_event(video_rows: list[dict], clip_length: int = CLIP_LENGTH) -> dict:
    """Pool frame-level scores across videos, including videos with no positives."""
    if not video_rows:
        raise ValueError("At least one video is required")
    videos = []
    pooled_labels = []
    pooled_scores = []
    for row in video_rows:
        summary = summarize_scores(row["frame_scores"], row["labels"], clip_length)
        count = summary["evaluated_frames"]
        pooled_labels.extend(row["labels"][:count])
        pooled_scores.extend(np.repeat(summary["clip_scores"], clip_length))
        videos.append({"video_id": row["video_id"], **summary})
    labels = np.asarray(pooled_labels, dtype=np.uint8)
    auc = float(roc_auc_score(labels, pooled_scores)) if len(np.unique(labels)) == 2 else None
    return {
        "video_count": len(videos),
        "total_frames": sum(video["total_frames"] for video in videos),
        "evaluated_frames": len(labels),
        "positive_evaluated_frames": int(labels.sum()),
        "negative_evaluated_frames": int(len(labels) - labels.sum()),
        "excluded_tail_frames": sum(video["excluded_tail_frames"] for video in videos),
        "excluded_positive_tail_frames": sum(video["excluded_positive_tail_frames"] for video in videos),
        "videos_with_defined_auc": sum(video["single_video_auc"] is not None for video in videos),
        "pooled_auc": auc,
        "videos": videos,
    }


def load_protocol() -> tuple[dict[str, str], list[str], str]:
    prompt_bytes = PROMPTS_FILE.read_bytes()
    prompt_data = json.loads(prompt_bytes)
    if set(prompt_data) != set(EVENTS) | {"description"}:
        raise ValueError("Prompt file must define exactly the five regular Avenue events")
    prompts = {event: prompt_data[event] for event in EVENTS}
    if not all(isinstance(text, str) and text.strip() for text in prompts.values()):
        raise ValueError("Each event prompt must be nonempty text")
    with VIDEO_GROUPS_FILE.open(encoding="utf-8") as file:
        groups = json.load(file)
    if set(groups) != set(EVENTS):
        raise ValueError("Official Avenue video groups do not match the five events")
    video_ids = sorted(set().union(*(groups[event] for event in EVENTS)))
    if len(video_ids) != 18:
        raise ValueError(f"Expected 18 regular videos, found {len(video_ids)}")
    import h5py
    for event in EVENTS:
        with h5py.File(LABEL_ROOT / f"avenue_{event}_label.h5", "r") as file:
            if set(file.keys()) != set(video_ids):
                raise ValueError(f"Label coverage differs for event {event}")
    return prompts, video_ids, hashlib.sha256(prompt_bytes).hexdigest()


def score_video(frames: list[Path], model, preprocess, text_features, device: str, batch_size: int) -> np.ndarray:
    import torch
    from PIL import Image

    batches = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            images = []
            for path in frames[start:start + batch_size]:
                with Image.open(path) as source:
                    images.append(preprocess(source.convert("RGB")))
            image_features = model.encode_image(torch.stack(images).to(device)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            batches.append((image_features @ text_features.T).cpu().numpy())
    return np.concatenate(batches, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    if not MODEL_PATH.is_file():
        parser.error(f"Local CLIP weights are missing: {MODEL_PATH}")
    prompts, video_ids, prompt_sha256 = load_protocol()

    import clip
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    model, preprocess = clip.load(str(MODEL_PATH), device=device, jit=False)
    model.eval()
    with torch.no_grad():
        tokens = clip.tokenize([prompts[event] for event in EVENTS], truncate=True).to(device)
        text_features = model.encode_text(tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    rows = {event: [] for event in EVENTS}
    for index, video_id in enumerate(video_ids, 1):
        frames = numbered_frames(FRAMES_ROOT / video_id)
        similarities = score_video(frames, model, preprocess, text_features, device, args.batch_size)
        for event_index, event in enumerate(EVENTS):
            labels = load_labels(LABEL_ROOT / f"avenue_{event}_label.h5", video_id, len(frames))
            rows[event].append({"video_id": video_id, "frame_scores": similarities[:, event_index].tolist(), "labels": labels.tolist()})
        print(f"{index}/{len(video_ids)} video {video_id}: {len(frames)} frames scored for five prompts", flush=True)

    results = {event: aggregate_event(rows[event]) for event in EVENTS}
    report = {
        "method": "CLIP-only diagnostic; maximum frame similarity per complete 24-frame clip",
        "model": "CLIP ViT-B/32",
        "prompt_config": str(PROMPTS_FILE.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "prompt_sha256": prompt_sha256,
        "prompts": prompts,
        "video_ids": video_ids,
        "device": device,
        "clip_length": CLIP_LENGTH,
        "tail_policy": "exclude incomplete clips from AUROC, matching the reference script",
        "aggregation": "frame-level AUROC pooled over all 18 regular videos per event; no per-video normalization",
        "events": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for event in EVENTS:
        result = results[event]
        auc = "unavailable" if result["pooled_auc"] is None else f"{result['pooled_auc']:.6f}"
        print(f"{event}: pooled AUROC={auc}, positives={result['positive_evaluated_frames']}, tail={result['excluded_tail_frames']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
