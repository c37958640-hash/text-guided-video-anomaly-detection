import importlib
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

def make_inputs(root: Path, frame_count: int = 50) -> tuple[Path, Path]:
    frame_dir = root / "frames" / "16"
    frame_dir.mkdir(parents=True)
    for index in range(1, frame_count + 1):
        (frame_dir / f"{index:06d}.jpg").touch()
    label_dir = root / "labels"
    label_dir.mkdir()
    labels = np.zeros(frame_count, dtype=np.uint8)
    labels[-2:] = 1
    with h5py.File(label_dir / "avenue_bicycle_label.h5", "w") as file:
        file.create_dataset("16", data=labels)
    return frame_dir.parent, label_dir


class AnyAnomalySingleVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            module = importlib.import_module("scripts.anyanomaly_qwen_single_video")
        except ModuleNotFoundError as exc:
            raise AssertionError(f"Single-video entry is missing: {exc}") from exc
        cls.prepare_video = staticmethod(module.prepare_video)
        cls.run_segments = staticmethod(module.run_segments)

    def test_preflight_counts_complete_segments_and_labeled_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frames_root, label_root = make_inputs(Path(directory))

            prepared = self.prepare_video("16", "bicycle", frames_root, label_root)

            self.assertEqual(prepared["total_frames"], 50)
            self.assertEqual(prepared["clip_count"], 2)
            self.assertEqual(prepared["evaluated_frames"], 48)
            self.assertEqual(prepared["excluded_tail_frames"], 2)
            self.assertEqual(prepared["excluded_positive_tail_frames"], 2)

    def test_interrupted_run_resumes_without_repeating_saved_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root, label_root = make_inputs(root)
            prepared = self.prepare_video("16", "bicycle", frames_root, label_root)
            output = root / "result.json"
            seen = []

            def interrupted(paths):
                seen.append((paths[0].name, paths[-1].name))
                if len(seen) == 2:
                    raise RuntimeError("inference stopped")
                return ["Score: 0.2", "Score: 0.4", "Score: 0.6"]

            with self.assertRaisesRegex(RuntimeError, "inference stopped"):
                self.run_segments(prepared, output, interrupted, model_id="local-qwen", clip_weights_id="local-clip")
            partial = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(partial["complete"])
            self.assertEqual(len(partial["segments"]), 1)
            self.assertEqual(partial["segments"][0]["answers"], ["Score: 0.2", "Score: 0.4", "Score: 0.6"])

            resumed = self.run_segments(
                prepared, output, lambda paths: ["0.3", "0.5", "0.7"],
                model_id="local-qwen", clip_weights_id="local-clip",
            )
            self.assertEqual(seen, [("000001.jpg", "000024.jpg"), ("000025.jpg", "000048.jpg")])
            self.assertTrue(resumed["complete"])
            self.assertEqual([(row["start_frame"], row["end_frame"]) for row in resumed["segments"]], [(1, 24), (25, 48)])
            self.assertEqual(resumed["scores"][:24], [0.2] * 24)
            self.assertEqual(resumed["scores"][-24:], [0.3] * 24)

    def test_rejects_out_of_range_answer_without_saving_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root, label_root = make_inputs(root)
            prepared = self.prepare_video("16", "bicycle", frames_root, label_root)
            output = root / "result.json"

            with self.assertRaisesRegex(ValueError, "0 to 1"):
                self.run_segments(
                    prepared, output, lambda paths: ["score: 2.0", "0.4", "0.6"],
                    model_id="local-qwen", clip_weights_id="local-clip",
                )
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["segments"], [])
            self.assertFalse(saved["complete"])

    def test_rejects_negative_answer_instead_of_reading_its_digits_as_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_root, label_root = make_inputs(root)
            prepared = self.prepare_video("16", "bicycle", frames_root, label_root)

            with self.assertRaisesRegex(ValueError, "0 to 1"):
                self.run_segments(
                    prepared, root / "result.json",
                    lambda paths: ["score: -0.2", "0.4", "0.6"],
                    model_id="local-qwen", clip_weights_id="local-clip",
                )


if __name__ == "__main__":
    unittest.main()
