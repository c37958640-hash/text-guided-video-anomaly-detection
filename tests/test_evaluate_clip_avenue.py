import importlib
import unittest


class AvenueClipBatchTests(unittest.TestCase):
    def test_pooled_auc_uses_frames_from_all_videos_including_all_negative_video(self) -> None:
        try:
            module = importlib.import_module("scripts.evaluate_clip_avenue")
        except ModuleNotFoundError as exc:
            self.fail(f"Batch evaluator is missing: {exc}")
        result = module.aggregate_event(
            [
                {"video_id": "01", "frame_scores": [0.1, 0.9, 0.3], "labels": [0, 1, 1]},
                {"video_id": "02", "frame_scores": [0.2, 0.4, 0.5], "labels": [0, 0, 0]},
            ],
            clip_length=2,
        )
        self.assertEqual(result["video_count"], 2)
        self.assertEqual(result["evaluated_frames"], 4)
        self.assertEqual(result["excluded_tail_frames"], 2)
        self.assertEqual(result["excluded_positive_tail_frames"], 1)
        self.assertIsNone(result["videos"][1]["single_video_auc"])
        self.assertAlmostEqual(result["pooled_auc"], 5 / 6)


if __name__ == "__main__":
    unittest.main()
