"""Reproduce a single-video, text-guided CLIP diagnostic on Avenue frames.

This is a CLIP-only check, not the paper's Qwen/vLLM baseline.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAMES_ROOT = PROJECT_ROOT / "datasets/avenue/testing/frames"
LABEL_ROOT = PROJECT_ROOT / "third_party/Paper-AnyAnomaly/ground_truth/c-avenue_labels"
MODEL_PATH = PROJECT_ROOT / "models/clip/ViT-B-32.pt"
EVENTS = ("bicycle", "dancing", "running", "throwing", "too_close")
CLIP_LENGTH = 24


def numbered_frames(frame_dir: Path) -> list[Path]:
    """Return JPG frames only when their one-based numbering has no gaps."""
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No JPG frames found in {frame_dir}")
    expected = [f"{index:06d}.jpg" for index in range(1, len(frames) + 1)]
    if [path.name for path in frames] != expected:
        raise ValueError(f"JPG numbering must run from 000001 without gaps: {frame_dir}")
    return frames


def load_labels(label_path: Path, video_id: str, frame_count: int) -> np.ndarray:
    if not label_path.is_file():
        raise FileNotFoundError(f"Label file is missing: {label_path}")
    with h5py.File(label_path, "r") as source:
        if video_id not in source:
            raise ValueError(f"Video {video_id} has no label in {label_path.name}")
        labels = np.asarray(source[video_id][:])
    if labels.ndim != 1 or len(labels) != frame_count:
        raise ValueError(f"Label length for video {video_id} does not match its {frame_count} frames")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")
    return labels.astype(np.uint8)


def summarize_scores(frame_scores: list[float], labels: list[int], clip_length: int = CLIP_LENGTH) -> dict:
    """Score complete clips and report exactly which tail frames are excluded."""
    scores = np.asarray(frame_scores, dtype=np.float64)
    truth = np.asarray(labels)
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")
    if scores.ndim != 1 or truth.ndim != 1 or len(scores) != len(truth):
        raise ValueError("Frame scores and labels must be one-dimensional and equally long")
    if not np.isfinite(scores).all() or not np.isin(truth, [0, 1]).all():
        raise ValueError("Scores must be finite and labels must be binary")
    evaluated = (len(scores) // clip_length) * clip_length
    if evaluated == 0:
        raise ValueError("The video has no complete clip")
    evaluated_labels = truth[:evaluated]
    clip_scores = scores[:evaluated].reshape(-1, clip_length).max(axis=1)
    evaluated_scores = np.repeat(clip_scores, clip_length)
    auc = float(roc_auc_score(evaluated_labels, evaluated_scores)) if len(np.unique(evaluated_labels)) == 2 else None
    clips = [
        {"clip_index": index + 1, "start_frame": index * clip_length + 1,
         "end_frame": (index + 1) * clip_length, "score": float(score)}
        for index, score in enumerate(clip_scores)
    ]
    return {
        "total_frames": len(scores),
        "evaluated_frames": evaluated,
        "excluded_tail_frames": len(scores) - evaluated,
        "excluded_positive_tail_frames": int(truth[evaluated:].sum()),
        "positive_evaluated_frames": int(evaluated_labels.sum()),
        "clip_count": len(clips),
        "clip_scores": clip_scores.tolist(),
        "clips": clips,
        "single_video_auc": auc,
    }


def frame_similarity(frames: list[Path], text: str, model_path: Path, batch_size: int, device: str) -> list[float]:
    if not text.strip():
        raise ValueError("Event text cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(f"Local CLIP weights are missing: {model_path}")

    import clip
    import torch
    from PIL import Image

    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model.eval()
    result = []
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text], truncate=True).to(device)).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for start in range(0, len(frames), batch_size):
            images = []
            for path in frames[start:start + batch_size]:
                with Image.open(path) as source:
                    images.append(preprocess(source.convert("RGB")))
            image_features = model.encode_image(torch.stack(images).to(device)).float()
            image_features /= image_features.norm(dim=-1, keepdim=True)
            result.extend((image_features @ text_features.T).squeeze(1).cpu().tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="16", help="Avenue test video number, e.g. 16")
    parser.add_argument("--text", default="a person riding a bicycle", help="English event description for CLIP")
    parser.add_argument("--event", choices=EVENTS, default="bicycle", help="Label category matching the text")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, help="JSON report path; defaults to outputs/clip_<video>_<event>.json")
    args = parser.parse_args()

    import torch
    video_number = int(args.video_id)
    if not 1 <= video_number <= 21:
        parser.error("video-id must be between 01 and 21")
    video_id = f"{video_number:02d}"
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    frames = numbered_frames(FRAMES_ROOT / video_id)
    labels = load_labels(LABEL_ROOT / f"avenue_{args.event}_label.h5", video_id, len(frames))
    scores = frame_similarity(frames, args.text, MODEL_PATH, args.batch_size, device)
    result = summarize_scores(scores, labels, CLIP_LENGTH)
    report = {
        "method": "CLIP-only diagnostic; maximum frame similarity per complete 24-frame clip",
        "model": "CLIP ViT-B/32",
        "video_id": video_id,
        "event": args.event,
        "query": args.text,
        "device": device,
        "clip_length": CLIP_LENGTH,
        "tail_policy": "exclude from AUROC, matching the reference script",
        **result,
        "frame_similarity": scores,
    }
    output = args.output or PROJECT_ROOT / f"outputs/clip_{video_id}_{args.event}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Video {video_id}: {result['total_frames']} frames, {result['clip_count']} complete clips")
    print(f"Excluded tail: {result['excluded_tail_frames']} frames, including {result['excluded_positive_tail_frames']} positives")
    if result["single_video_auc"] is None:
        print("Single-video AUROC: unavailable (only one label class in evaluated frames)")
    else:
        print(f"Single-video AUROC: {result['single_video_auc']:.6f} (CLIP-only diagnostic)")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
