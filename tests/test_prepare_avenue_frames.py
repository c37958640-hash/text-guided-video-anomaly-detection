import tempfile
import unittest
from pathlib import Path

import cv2
import h5py
import numpy as np

from scripts.prepare_avenue_frames import audit_label_files, prepare_video


def make_video(path: Path, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (64, 48))
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    for index in range(frame_count):
        frame = np.full((48, 64, 3), index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class PrepareAvenueFramesTests(unittest.TestCase):
    def test_extracts_ordered_frames_and_reports_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "01.avi"
            make_video(video, 26)
            frames_root = root / "frames"

            result = prepare_video(video, frames_root, clip_length=24)

            self.assertEqual(result["frames"], 26)
            self.assertEqual(result["full_clips"], 1)
            self.assertEqual(result["tail_frames"], 2)
            self.assertEqual(result["status"], "created")
            self.assertEqual(
                [path.name for path in sorted((frames_root / "01").glob("*.jpg"))],
                [f"{index:06d}.jpg" for index in range(1, 27)],
            )
            self.assertIsNotNone(cv2.imread(str(frames_root / "01" / "000026.jpg")))
            self.assertEqual(prepare_video(video, frames_root)["status"], "verified_existing")

    def test_rejects_incomplete_existing_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "01.avi"
            make_video(video, 3)
            frames_root = root / "frames"
            prepare_video(video, frames_root)
            (frames_root / "01" / "000002.jpg").unlink()

            with self.assertRaises(RuntimeError):
                prepare_video(video, frames_root)

    def test_label_audit_rejects_wrong_frame_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            label_dir = Path(directory)
            with h5py.File(label_dir / "avenue_running_label.h5", "w") as labels:
                labels.create_dataset("01", data=np.zeros(25, dtype=np.uint8))

            with self.assertRaises(ValueError):
                audit_label_files(label_dir, {"01": 26})


if __name__ == "__main__":
    unittest.main()
