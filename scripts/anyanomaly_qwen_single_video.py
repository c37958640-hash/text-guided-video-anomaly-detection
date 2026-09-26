"""Run one regular Avenue video with the reference AnyAnomaly Qwen method.

The default command only checks local frames and labels. --run requires local
model files and a Linux GPU; it never uses a remote model name.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable

from scripts.clip_single_video_diagnostic import (
    CLIP_LENGTH,
    EVENTS,
    FRAMES_ROOT,
    LABEL_ROOT,
    MODEL_PATH,
    PROJECT_ROOT,
    load_labels,
    numbered_frames,
)

REFERENCE_ROOT = PROJECT_ROOT / "third_party/Paper-AnyAnomaly"
REGULAR_VIDEO_IDS = {f"{number:02d}" for number in range(1, 22)} - {"06", "09", "11"}
SCORE_NAMES = ("scores", "scores_wa", "scores_tc")


def prepare_video(
    video_id: str, event: str, frames_root: Path = FRAMES_ROOT,
    label_root: Path = LABEL_ROOT,
) -> dict:
    """Check one video's frame order and matching event labels without a model."""
    if event not in EVENTS:
        raise ValueError(f"Unknown Avenue event: {event}")
    try:
        normalized_id = f"{int(video_id):02d}"
    except ValueError as exc:
        raise ValueError("Video ID must be a number from 01 to 21") from exc
    if normalized_id not in REGULAR_VIDEO_IDS:
        raise ValueError(f"Video {normalized_id} is outside the 18 regular Avenue videos")
    frames = numbered_frames(frames_root / normalized_id)
    labels = load_labels(label_root / f"avenue_{event}_label.h5", normalized_id, len(frames))
    complete = len(frames) // CLIP_LENGTH
    if complete == 0:
        raise ValueError("The video has no complete 24-frame segment")
    evaluated = complete * CLIP_LENGTH
    return {
        "video_id": normalized_id,
        "event": event,
        "frames": frames,
        "total_frames": len(frames),
        "clip_length": CLIP_LENGTH,
        "clip_count": complete,
        "evaluated_frames": evaluated,
        "excluded_tail_frames": len(frames) - evaluated,
        "excluded_positive_tail_frames": int(labels[evaluated:].sum()),
        "positive_evaluated_frames": int(labels[:evaluated].sum()),
    }


def _score_from_answer(answer: str) -> float:
    # Use the first number as the reference does; keep its sign for validation.
    match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", answer)
    if match is None:
        raise ValueError("Qwen answer has no numeric score")
    score = float(match.group(0))
    if not 0.0 <= score <= 1.0:
        raise ValueError("Qwen score must be from 0 to 1")
    return score


def _write_checkpoint(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)


def _update_scores(report: dict) -> None:
    for channel, position in zip(SCORE_NAMES, range(3)):
        report[channel] = [
            segment["segment_scores"][position]
            for segment in report["segments"]
            for _ in range(report["clip_length"])
        ]
    report["complete"] = len(report["segments"]) == report["clip_count"]


def run_segments(
    prepared: dict,
    output: Path,
    scorer: Callable[[list[Path]], list[str]],
    *,
    model_id: str,
    clip_weights_id: str,
) -> dict:
    """Score full clips, atomically checkpoint each success, and resume safely."""
    frames = prepared["frames"]
    identity = {
        "video_id": prepared["video_id"],
        "event": prepared["event"],
        "model_id": model_id,
        "clip_weights_id": clip_weights_id,
        "frame_names_sha256": hashlib.sha256(
            "\n".join(path.name for path in frames).encode("utf-8")
        ).hexdigest(),
    }
    public = {key: value for key, value in prepared.items() if key != "frames"}
    if output.exists():
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("identity") != identity or any(report.get(key) != value for key, value in public.items()):
            raise ValueError(f"Existing checkpoint belongs to a different run: {output}")
        if not isinstance(report.get("segments"), list) or len(report["segments"]) > prepared["clip_count"]:
            raise ValueError(f"Invalid checkpoint segments: {output}")
        for index, segment in enumerate(report["segments"]):
            if segment.get("clip_index") != index + 1:
                raise ValueError(f"Checkpoint segment order is invalid: {output}")
    else:
        report = {**public, "identity": identity, "segments": [], "last_error": None}
        _update_scores(report)
        _write_checkpoint(output, report)

    for index in range(len(report["segments"]), prepared["clip_count"]):
        start = index * prepared["clip_length"]
        paths = frames[start:start + prepared["clip_length"]]
        answers = scorer(paths)
        if not isinstance(answers, (tuple, list)) or len(answers) != 3 or not all(
            isinstance(answer, str) for answer in answers
        ):
            raise ValueError("Qwen must return three text answers per segment")
        try:
            scores = [_score_from_answer(answer) for answer in answers]
        except ValueError as exc:
            report["last_error"] = {
                "clip_index": index + 1,
                "answers": list(answers),
                "message": str(exc),
            }
            _write_checkpoint(output, report)
            raise
        report["segments"].append({
            "clip_index": index + 1,
            "start_frame": start + 1,
            "end_frame": start + prepared["clip_length"],
            "answers": list(answers),
            "segment_scores": scores,
        })
        report["last_error"] = None
        _update_scores(report)
        _write_checkpoint(output, report)
    return report


def _reference_scorer(event: str, frames_root: Path, label_root: Path,
                      reference_root: Path, model_path: Path, clip_weights: Path):
    """Load the official preprocessing and Qwen functions only for a real run."""
    if os.name == "nt":
        raise RuntimeError("Qwen/vLLM inference needs Linux or WSL; Windows can run the frame/label check")
    if not reference_root.joinpath("vad_proposed_Qwen_vllm.py").is_file():
        raise FileNotFoundError(f"Official reference code is missing: {reference_root}")
    if not model_path.is_dir() or not model_path.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Local Qwen model files are missing: {model_path}")
    if not clip_weights.is_file():
        raise FileNotFoundError(f"Local CLIP weights are missing: {clip_weights}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    sys.path.insert(0, str(reference_root))
    import clip
    import torch
    # The reference config creates a results directory during import.
    previous_directory = Path.cwd()
    try:
        os.chdir(reference_root)
        import config as reference_config
    finally:
        os.chdir(previous_directory)
    from functions.attn_func import winclip_attention
    from functions.grid_func import grid_generation
    from functions.key_func import KFS
    from functions.qwen_vllm_func import load_lvlm, lvlm_test, make_instruction, qwen_make_messages
    from functions.text_func import make_text_embedding

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this Qwen/vLLM run")
    device = torch.device("cuda")
    reference_config.share_config["data_root"] = str(frames_root.parent.parent.parent)
    reference_config.share_config["cdata_root"] = str(label_root.parent)
    settings = argparse.Namespace(
        dataset="avenue", type=event, multiple=False, prompt_type=3,
        anomaly_detect=True, calc_auc=False, calc_video_auc=False,
        clip_length=CLIP_LENGTH, template_adaption=True, class_adaption=True,
        kfs_num=4, lge_scale=True, mid_scale=True, sml_scale=True, stride=False,
        model_path=str(model_path), grid_search=False, sigma_range="1,30",
        weight_step=0.1, sigma=15, alpha=0.6, beta=0.3, gamma=0.1,
    )
    cfg = reference_config.update_config(settings)
    model, processor, sampling = load_lvlm(str(model_path))
    clip_model, preprocess = clip.load(str(clip_weights), device=device)
    key_selector = KFS(cfg.kfs_num, cfg.clip_length, clip_model, preprocess, device)
    instruction, temporal_instruction = make_instruction(cfg, event)
    text_embedding = make_text_embedding(
        clip_model, device, text=event, type_list=cfg.type_list,
        class_adaption=cfg.class_adaption, template_adaption=cfg.template_adaption,
    )

    def score(paths: list[Path]) -> list[str]:
        clip_paths = [str(path) for path in paths]
        indices = key_selector.call_function(clip_paths, event)
        main_key = clip_paths[indices[0]]
        other_keys = [clip_paths[position] for position in indices[1:]]
        attention_image = winclip_attention(
            cfg, main_key, text_embedding, clip_model, device,
            cfg.class_adaption, cfg.type_ids[0],
        )
        temporal_image = grid_generation(cfg, other_keys, event, clip_model, device)
        messages = [
            qwen_make_messages(main_key, instruction),
            qwen_make_messages(attention_image, instruction),
            qwen_make_messages(temporal_image, temporal_instruction),
        ]
        return lvlm_test(model, processor, sampling, messages)

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", default="16")
    parser.add_argument("--event", choices=EVENTS, default="bicycle")
    parser.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--run", action="store_true", help="Run Qwen on Linux using local model files")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--qwen-model", type=Path, help="Existing local Qwen model directory; no download")
    parser.add_argument("--clip-weights", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepared = prepare_video(args.video_id, args.event, args.frames_root, args.label_root)
    print(
        f"Video {prepared['video_id']}, event {args.event}: {prepared['total_frames']} frames, "
        f"{prepared['clip_count']} full segments, {prepared['excluded_tail_frames']} tail frames"
    )
    if not args.run:
        output = args.output or PROJECT_ROOT / f"outputs/anyanomaly/preflight_{prepared['video_id']}_{args.event}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({key: value for key, value in prepared.items() if key != "frames"},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Frame/label check saved: {output}")
        return
    if args.qwen_model is None:
        parser.error("--run requires --qwen-model pointing to an existing local model")
    output = args.output or PROJECT_ROOT / f"outputs/anyanomaly/single_{prepared['video_id']}_{args.event}.json"
    scorer = _reference_scorer(
        args.event, args.frames_root, args.label_root, args.reference_root,
        args.qwen_model, args.clip_weights,
    )
    result = run_segments(
        prepared, output, scorer,
        model_id=str(args.qwen_model.resolve()),
        clip_weights_id=str(args.clip_weights.resolve()),
    )
    print(f"Saved {len(result['segments'])}/{prepared['clip_count']} segments: {output}")


if __name__ == "__main__":
    main()
