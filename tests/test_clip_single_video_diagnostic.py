import importlib
import unittest


class ClipSingleVideoDiagnosticTests(unittest.TestCase):
    def test_full_clips_use_max_score_and_exclude_labeled_tail(self) -> None:
        try:
            module = importlib.import_module("scripts.clip_single_video_diagnostic")
        except ModuleNotFoundError as exc:
            self.fail(f"Diagnostic script is missing: {exc}")

        result = module.summarize_scores(
            frame_scores=[0.1, 0.9, 0.2, 0.8, 0.3],
            labels=[0, 1, 0, 0, 1],
            clip_length=2,
        )
        self.assertEqual(result["clip_scores"], [0.9, 0.8])
        self.assertEqual(result["evaluated_frames"], 4)
        self.assertEqual(result["excluded_tail_frames"], 1)
        self.assertEqual(result["excluded_positive_tail_frames"], 1)
        self.assertEqual(
            [(row["start_frame"], row["end_frame"]) for row in result["clips"]],
            [(1, 2), (3, 4)],
        )
        self.assertAlmostEqual(result["single_video_auc"], 5 / 6)


    def test_all_negative_video_still_reports_scores_without_auc(self) -> None:
        module = importlib.import_module("scripts.clip_single_video_diagnostic")
        try:
            result = module.summarize_scores(
                frame_scores=[0.1, 0.9, 0.2, 0.8],
                labels=[0, 0, 0, 0],
                clip_length=2,
            )
        except ValueError as exc:
            self.fail(f"Scores should still be available: {exc}")
        self.assertEqual(result["clip_scores"], [0.9, 0.8])
        self.assertIsNone(result["single_video_auc"])



if __name__ == "__main__":
    unittest.main()
