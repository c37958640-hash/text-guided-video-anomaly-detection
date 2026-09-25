"""Convert the official CUHK Avenue test videos to AnyAnomaly's JPG layout."""

import json
from pathlib import Path

import cv2
import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_VIDEOS = PROJECT_ROOT / "datasets/avenue/raw/Avenue Dataset/testing_videos"
FRAMES_ROOT = PROJECT_ROOT / "datasets/avenue/testing/frames"
LABEL_ROOT = PROJECT_ROOT / "third_party/Paper-AnyAnomaly/ground_truth/c-avenue_labels"
REPORT = PROJECT_ROOT / "outputs/avenue_frame_audit.json"
CLIP_LENGTH = 24


def prepare_video(video_path: Path, frames_root: Path, clip_length: int = CLIP_LENGTH) -> dict:
    """Decode one AVI into numbered JPGs without replacing existing output."""
    if clip_length <= 0:
        raise ValueError("clip_length must be positive")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if expected <= 0:
            raise RuntimeError(f"Video reports no frames: {video_path}")
        frames_root.mkdir(parents=True, exist_ok=True)
        target = frames_root / video_path.stem
        partial = frames_root / f".{video_path.stem}.partial"
        if partial.exists():
            raise RuntimeError(f"Incomplete extraction exists; inspect it first: {partial}")
        if target.exists():
            names = sorted(path.name for path in target.glob("*.jpg"))
            wanted = [f"{index:06d}.jpg" for index in range(1, expected + 1)]
            if names != wanted:
                raise RuntimeError(f"Existing frame directory is incomplete: {target}")
            if cv2.imread(str(target / wanted[0])) is None or cv2.imread(str(target / wanted[-1])) is None:
                raise RuntimeError(f"Existing first or last JPG is unreadable: {target}")
            status = "verified_existing"
            count = expected
        else:
            partial.mkdir()
            count = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                count += 1
                image_path = partial / f"{count:06d}.jpg"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not write JPG: {image_path}")
            if count != expected:
                raise RuntimeError(f"Decoded {count} frames, video reports {expected}: {video_path}")
            partial.rename(target)
            status = "created"
        return {
            "video_id": video_path.stem,
            "frames": count,
            "fps": fps,
            "full_clips": count // clip_length,
            "tail_frames": count % clip_length,
            "status": status,
        }
    finally:
        cap.release()


def audit_label_files(label_dir: Path, frame_counts: dict[str, int]) -> dict:
    """Check that each HDF5 label array matches its source video's frames."""
    files = sorted(label_dir.glob("*.h5"))
    if not files:
        raise RuntimeError(f"No label files found: {label_dir}")
    summary = {}
    for path in files:
        with h5py.File(path, "r") as labels:
            keys = sorted(labels.keys())
            for video_id in keys:
                if video_id not in frame_counts:
                    raise ValueError(f"Label {path.name} names missing video {video_id}")
                actual = len(labels[video_id])
                expected = frame_counts[video_id]
                if actual != expected:
                    raise ValueError(
                        f"Label {path.name}/{video_id} has {actual} frames, video has {expected}"
                    )
            summary[path.name] = {
                "video_ids": keys,
                "videos": len(keys),
                "frames": sum(frame_counts[key] for key in keys),
            }
    return summary


def main() -> None:
    videos = sorted(RAW_VIDEOS.glob("*.avi"))
    expected_ids = [f"{index:02d}" for index in range(1, 22)]
    if [path.stem for path in videos] != expected_ids:
        raise RuntimeError(f"Expected Avenue test videos 01..21 in {RAW_VIDEOS}")

    rows = []
    for video in videos:
        row = prepare_video(video, FRAMES_ROOT)
        rows.append(row)
        print(
            f"{row['video_id']}: {row['frames']} frames, "
            f"{row['full_clips']} full clips, {row['tail_frames']} tail frames "
            f"({row['status']})",
            flush=True,
        )

    frame_counts = {row["video_id"]: row["frames"] for row in rows}
    labels = audit_label_files(LABEL_ROOT, frame_counts)
    single_files = sorted(path.name for path in LABEL_ROOT.glob("avenue_*_label.h5"))
    if len(single_files) != 5:
        raise RuntimeError(f"Expected five single-event label files, found {len(single_files)}")
    single_ids = [set(labels[name]["video_ids"]) for name in single_files]
    if len(set(map(frozenset, single_ids))) != 1:
        raise RuntimeError("Five single-event label files do not cover the same videos")
    multi_files = sorted(path.name for path in LABEL_ROOT.glob("m_avenue_*_label.h5"))
    if len(multi_files) != 2:
        raise RuntimeError(f"Expected two multi-event label files, found {len(multi_files)}")
    multi_ids = set().union(*(labels[name]["video_ids"] for name in multi_files))
    if single_ids[0] & multi_ids or single_ids[0] | multi_ids != set(frame_counts):
        raise RuntimeError("Single-event and multi-event labels do not partition the 21 videos")

    report = {
        "clip_length": CLIP_LENGTH,
        "videos": rows,
        "labels": labels,
        "single_event_video_ids": sorted(single_ids[0]),
        "multi_event_video_ids": sorted(multi_ids),
        "summary": {
            "all_videos": len(rows),
            "all_frames": sum(row["frames"] for row in rows),
            "single_event_videos": len(single_ids[0]),
            "single_event_frames": sum(frame_counts[key] for key in single_ids[0]),
            "single_event_full_clips": sum(row["full_clips"] for row in rows if row["video_id"] in single_ids[0]),
            "single_event_tail_frames": sum(row["tail_frames"] for row in rows if row["video_id"] in single_ids[0]),
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Audit report: {REPORT}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
