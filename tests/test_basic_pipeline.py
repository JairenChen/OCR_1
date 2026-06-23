from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.batch_runner import (
    BatchVideoResult,
    build_video_output_dir,
    copy_srt_to_video_dir,
    discover_video_files,
    export_batch_summary_json,
    export_batch_summary_txt,
    parse_video_extensions,
)
from src.boundary_refiner import refine_segments_boundaries
from src.exporters import export_segments_json
from src.models import FrameOcrResult, OcrLine, SampledFrame, SubtitleSegment
from src.ocr_cache import clone_ocr_result_for_frame, frame_difference_metrics, make_frame_fingerprint
from src.ocr_engine import PaddleOCREngine, payload_to_ocr_lines
from src.run_report import build_run_report, export_run_report_json, export_run_report_txt, format_run_report_text
from src.subtitle_merger import (
    box_iou,
    line_match_score,
    merge_repeated_text_snapshots,
    merge_text_tracks_state_machine,
)
from src.text_filter import frame_text, get_filter_reason, get_noise_filter_reason, sort_ocr_lines_by_layout
from src.utils import format_timestamp
from src.video_reader import VideoInfo


class BasicPipelineTest(unittest.TestCase):
    def test_format_timestamp(self) -> None:
        self.assertEqual(format_timestamp(1234), "00:00:01.234")
        self.assertEqual(format_timestamp(1234, srt=True), "00:00:01,234")

    def test_batch_video_discovery_output_dir_and_srt_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_dir = temp_path / "videos"
            nested_dir = video_dir / "nested"
            nested_dir.mkdir(parents=True)
            first_video = video_dir / "first.mp4"
            second_video = nested_dir / "second.mkv"
            ignored_file = video_dir / "notes.txt"
            first_video.write_bytes(b"fake video")
            second_video.write_bytes(b"fake video")
            ignored_file.write_text("ignore", encoding="utf-8")

            extensions = parse_video_extensions("mp4,mkv")
            flat_videos = discover_video_files(video_dir, extensions=extensions, recursive=False)
            recursive_videos = discover_video_files(video_dir, extensions=extensions, recursive=True)
            self.assertEqual(flat_videos, [first_video])
            self.assertEqual(recursive_videos, [first_video, second_video])

            output_dir = build_video_output_dir(
                video_path=second_video,
                video_root=video_dir,
                output_root=temp_path / "outputs",
            )
            self.assertEqual(output_dir, temp_path / "outputs" / "nested" / "second")

            generated_srt = output_dir / "second.srt"
            generated_srt.parent.mkdir(parents=True)
            generated_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
            copied_srt = copy_srt_to_video_dir(generated_srt, second_video)
            self.assertEqual(copied_srt, nested_dir / "second.srt")
            self.assertTrue(copied_srt.exists())

            result = BatchVideoResult(
                video_path=second_video,
                output_dir=output_dir,
                frames_dir=output_dir / "frames",
                status="success",
                written_paths=[generated_srt],
                copied_srt_path=copied_srt,
                elapsed_seconds=1.23,
            )
            json_path = export_batch_summary_json(temp_path / "summary.json", [result], 1.23)
            txt_path = export_batch_summary_txt(temp_path / "summary.txt", [result], 1.23)
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())

    def test_parse_paddle_payload(self) -> None:
        payload = {
            "rec_texts": ["hello", "", "world"],
            "rec_scores": [0.9, 0.1, 0.8],
            "rec_boxes": [[0, 0, 10, 10], [0, 0, 1, 1], [10, 10, 20, 20]],
        }
        lines = payload_to_ocr_lines(payload, score_threshold=0.5)
        self.assertEqual([line.text for line in lines], ["hello", "world"])
        self.assertEqual(lines[0].box, [0.0, 0.0, 10.0, 10.0])

    def test_ocr_cache_frame_difference_and_clone(self) -> None:
        import numpy as np

        first = np.zeros((48, 64, 3), dtype=np.uint8)
        second = first.copy()
        second[10:20, 20:30] = 255

        first_fingerprint = make_frame_fingerprint(first, image_size=16)
        same_fingerprint = make_frame_fingerprint(first.copy(), image_size=16)
        changed_fingerprint = make_frame_fingerprint(second, image_size=16)

        same_metrics = frame_difference_metrics(first_fingerprint, same_fingerprint)
        changed_metrics = frame_difference_metrics(first_fingerprint, changed_fingerprint)
        self.assertEqual(same_metrics.mean, 0.0)
        self.assertEqual(same_metrics.maximum, 0.0)
        self.assertGreater(changed_metrics.mean, 0.0)
        self.assertGreater(changed_metrics.maximum, 0.0)
        self.assertTrue(same_metrics.is_reusable(mean_threshold=0.5, max_threshold=12.0))
        self.assertFalse(changed_metrics.is_reusable(mean_threshold=0.5, max_threshold=12.0))

        source = FrameOcrResult(
            frame_index=0,
            timestamp_ms=0,
            image_path=Path("previous.jpg"),
            width=64,
            height=48,
            lines=[OcrLine("hello", 0.9, polygon=[[0, 0], [10, 0]], box=[0, 0, 10, 10])],
        )
        cloned = clone_ocr_result_for_frame(
            source,
            SampledFrame(
                index=1,
                timestamp_ms=500,
                image_path=Path("current.jpg"),
                width=64,
                height=48,
            ),
        )
        self.assertEqual(cloned.frame_index, 1)
        self.assertEqual(cloned.timestamp_ms, 500)
        self.assertEqual(cloned.image_path, Path("current.jpg"))
        self.assertEqual(cloned.lines[0].text, "hello")
        source.lines[0].box[0] = 99
        self.assertEqual(cloned.lines[0].box[0], 0)

    def test_predict_frames_splits_batch_results(self) -> None:
        class FakeOCR:
            def predict(self, inputs, **kwargs):
                return [
                    {
                        "res": {
                            "rec_texts": [f"text-{index}"],
                            "rec_scores": [0.9],
                            "rec_boxes": [[0, 0, 10, 10]],
                        }
                    }
                    for index, _ in enumerate(inputs)
                ]

        engine = PaddleOCREngine(text_score_threshold=0.5)
        engine._ocr = FakeOCR()
        frames = [
            SampledFrame(index=0, timestamp_ms=0, image_path=Path("frame0.jpg"), image=object()),
            SampledFrame(index=1, timestamp_ms=500, image_path=Path("frame1.jpg"), image=object()),
        ]
        results = engine.predict_frames(frames)
        self.assertEqual([result.timestamp_ms for result in results], [0, 500])
        self.assertEqual([result.lines[0].text for result in results], ["text-0", "text-1"])

    def test_ocr_engine_passes_predict_options(self) -> None:
        class FakeOCR:
            def __init__(self) -> None:
                self.kwargs = None

            def predict(self, image, **kwargs):
                self.kwargs = kwargs
                return [
                    {
                        "res": {
                            "rec_texts": ["hello"],
                            "rec_scores": [0.9],
                            "rec_boxes": [[0, 0, 10, 10]],
                        }
                    }
                ]

        fake_ocr = FakeOCR()
        engine = PaddleOCREngine(
            text_det_limit_side_len=1280,
            text_det_limit_type="max",
            text_det_thresh=0.25,
            text_det_box_thresh=0.45,
            text_det_unclip_ratio=2.0,
            text_rec_score_thresh=0.0,
            return_word_box=False,
        )
        engine._ocr = fake_ocr
        lines = engine.predict_image_data(object())
        self.assertEqual(lines[0].text, "hello")
        self.assertEqual(
            fake_ocr.kwargs,
            {
                "text_det_limit_side_len": 1280,
                "text_det_limit_type": "max",
                "text_det_thresh": 0.25,
                "text_det_box_thresh": 0.45,
                "text_det_unclip_ratio": 2.0,
                "text_rec_score_thresh": 0.0,
                "return_word_box": False,
            },
        )

    def test_merge_repeated_text_snapshots(self) -> None:
        frames = [
            FrameOcrResult(0, 0, Path("frame0.jpg"), [OcrLine("hello", 0.9, box=[0, 0, 10, 10])]),
            FrameOcrResult(1, 1000, Path("frame1.jpg"), [OcrLine("hello", 0.8, box=[1, 1, 11, 11])]),
            FrameOcrResult(2, 2000, Path("frame2.jpg"), [OcrLine("next", 0.7, box=[0, 30, 10, 40])]),
        ]
        segments = merge_repeated_text_snapshots(frames, frame_interval_ms=1000)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "hello")
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 2000)

    def test_position_aware_merge_splits_same_text_at_different_position(self) -> None:
        frames = [
            FrameOcrResult(0, 0, Path("frame0.jpg"), [OcrLine("LIMIT", 0.9, box=[0, 0, 20, 10])]),
            FrameOcrResult(1, 1000, Path("frame1.jpg"), [OcrLine("LIMIT", 0.9, box=[80, 80, 100, 90])]),
        ]
        segments = merge_repeated_text_snapshots(
            frames,
            frame_interval_ms=1000,
            use_position=True,
            position_match_threshold=0.5,
            line_center_distance_threshold=0.05,
        )
        self.assertEqual(len(segments), 2)

    def test_text_only_merge_can_merge_same_text_at_different_position(self) -> None:
        frames = [
            FrameOcrResult(0, 0, Path("frame0.jpg"), [OcrLine("LIMIT", 0.9, box=[0, 0, 20, 10])]),
            FrameOcrResult(1, 1000, Path("frame1.jpg"), [OcrLine("LIMIT", 0.9, box=[80, 80, 100, 90])]),
        ]
        segments = merge_repeated_text_snapshots(
            frames,
            frame_interval_ms=1000,
            use_position=False,
        )
        self.assertEqual(len(segments), 1)

    def test_line_state_machine_bridges_short_miss(self) -> None:
        frames = [
            FrameOcrResult(0, 0, Path("frame0.jpg"), [OcrLine("LIMIT", 0.9, box=[0, 0, 20, 10])]),
            FrameOcrResult(1, 500, Path("frame1.jpg"), []),
            FrameOcrResult(2, 1000, Path("frame2.jpg"), [OcrLine("LIMIT", 0.8, box=[1, 0, 21, 10])]),
        ]
        segments = merge_text_tracks_state_machine(
            frames,
            frame_interval_ms=500,
            max_gap_ms=1000,
            use_position=True,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "LIMIT")
        self.assertEqual(segments[0].start_ms, 0)
        self.assertEqual(segments[0].end_ms, 1500)

    def test_line_state_machine_tracks_multiple_text_blocks(self) -> None:
        frames = [
            FrameOcrResult(
                0,
                0,
                Path("frame0.jpg"),
                [
                    OcrLine("TITLE", 0.9, box=[0, 0, 20, 10]),
                    OcrLine("$19", 0.8, box=[0, 40, 20, 50]),
                ],
            ),
            FrameOcrResult(
                1,
                500,
                Path("frame1.jpg"),
                [
                    OcrLine("TITLE", 0.9, box=[0, 0, 20, 10]),
                    OcrLine("$19", 0.9, box=[0, 40, 20, 50]),
                ],
            ),
        ]
        segments = merge_text_tracks_state_machine(frames, frame_interval_ms=500, max_gap_ms=750)
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment.text for segment in segments], ["TITLE", "$19"])
        self.assertEqual([segment.end_ms for segment in segments], [1000, 1000])

    def test_line_state_machine_splits_same_text_at_different_position(self) -> None:
        frames = [
            FrameOcrResult(0, 0, Path("frame0.jpg"), [OcrLine("LIMIT", 0.9, box=[0, 0, 20, 10])]),
            FrameOcrResult(1, 500, Path("frame1.jpg"), [OcrLine("LIMIT", 0.9, box=[80, 80, 100, 90])]),
        ]
        segments = merge_text_tracks_state_machine(
            frames,
            frame_interval_ms=500,
            max_gap_ms=750,
            use_position=True,
            line_center_distance_threshold=0.05,
        )
        self.assertEqual(len(segments), 2)

    def test_box_iou_and_line_match_score(self) -> None:
        self.assertGreater(box_iou([0, 0, 10, 10], [1, 1, 11, 11]), 0.5)
        score = line_match_score(
            [OcrLine("LIMIT", 0.9, box=[0, 0, 10, 10])],
            [OcrLine("LIMIT", 0.9, box=[1, 1, 11, 11])],
            frame_width=100,
            frame_height=100,
        )
        self.assertEqual(score, 1.0)

    def test_export_segments_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segments.json"
            export_segments_json(
                output_path,
                [SubtitleSegment(start_ms=0, end_ms=1500, text="hello", confidence=0.9)],
            )
            self.assertTrue(output_path.exists())
            self.assertIn("hello", output_path.read_text(encoding="utf-8"))

    def test_build_and_export_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "sample.mp4"
            video_path.write_bytes(b"fake video")
            video_info = VideoInfo(
                path=video_path,
                fps=25.0,
                frame_count=50,
                duration_ms=2000,
                width=640,
                height=360,
            )
            sampled_frames = [
                FrameOcrResult(0, 0, temp_path / "frame0.jpg", [OcrLine("hello", 0.9, box=[0, 0, 10, 10])]),
                FrameOcrResult(1, 1000, temp_path / "frame1.jpg", [OcrLine("world", 0.8, box=[0, 20, 10, 30])]),
            ]
            report = build_run_report(
                video_info=video_info,
                sampled_frames=[
                    SampledFrame(
                        index=0,
                        timestamp_ms=0,
                        image_path=temp_path / "frame0.jpg",
                    ),
                    SampledFrame(
                        index=1,
                        timestamp_ms=1000,
                        image_path=temp_path / "frame1.jpg",
                    ),
                ],
                frame_results=sampled_frames,
                segments=[SubtitleSegment(start_ms=0, end_ms=2000, text="hello\nworld", confidence=0.85)],
                written_paths=[temp_path / "sample.srt"],
                timings={
                    "sampling_seconds": 0.1,
                    "ocr_filter_debug_seconds": 0.2,
                    "merge_seconds": 0.01,
                    "export_seconds": 0.01,
                    "total_seconds": 0.32,
                },
                settings={
                    "command": "python -m src.main",
                    "config_path": "configs/default.yaml",
                    "sample_only": False,
                    "sampling": {"interval_seconds": 1.0, "max_frames": None},
                    "ocr": {"device": "gpu:0"},
                },
            )
            self.assertEqual(report["sampling"]["sampled_frames"], 2)
            self.assertEqual(report["ocr"]["total_lines"], 2)
            self.assertEqual(report["merge"]["segments"], 1)
            self.assertIn("总用时", format_run_report_text(report))

            json_path = export_run_report_json(temp_path / "report.json", report)
            txt_path = export_run_report_txt(temp_path / "report.txt", report)
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())

    def test_filter_reason_for_region(self) -> None:
        line = OcrLine("subtitle", 0.9, box=[10, 10, 20, 20])
        reason = get_filter_reason(
            line,
            min_confidence=0.5,
            frame_width=100,
            frame_height=100,
            subtitle_region={"enabled": True, "x_min": 0.0, "y_min": 0.5, "x_max": 1.0, "y_max": 1.0},
        )
        self.assertEqual(reason, "outside_region")

    def test_noise_filter_reason(self) -> None:
        config = {
            "enabled": True,
            "min_text_length": 2,
            "drop_symbol_only": True,
            "drop_repeated_punctuation": True,
            "drop_numeric_only": False,
        }
        self.assertEqual(get_noise_filter_reason("*", config), "noise_too_short")
        self.assertEqual(get_noise_filter_reason("++", config), "noise_symbol_only")
        self.assertIsNone(get_noise_filter_reason("50%", config))
        self.assertIsNone(get_noise_filter_reason("OK", config))

    def test_filter_reason_for_noise(self) -> None:
        reason = get_filter_reason(
            OcrLine("*", 0.9, box=[0, 0, 10, 10]),
            min_confidence=0.5,
            noise_config={"enabled": True, "min_text_length": 2},
        )
        self.assertEqual(reason, "noise_too_short")

    def test_sort_ocr_lines_by_layout(self) -> None:
        lines = [
            OcrLine("bottom right", 0.9, box=[50, 50, 90, 60]),
            OcrLine("top left", 0.9, box=[10, 10, 40, 20]),
            OcrLine("bottom left", 0.9, box=[10, 50, 40, 60]),
            OcrLine("top right", 0.9, box=[50, 10, 90, 20]),
        ]
        ordered = sort_ocr_lines_by_layout(lines)
        self.assertEqual(
            [line.text for line in ordered],
            ["top left", "top right", "bottom left", "bottom right"],
        )
        self.assertEqual(frame_text(lines), "top left\ntop right\nbottom left\nbottom right")

    def test_draw_debug_image(self) -> None:
        cv2 = __import__("cv2")
        import numpy as np

        from src.debug_visualizer import draw_debug_image

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "frame.jpg"
            image = np.zeros((60, 80, 3), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)
            frame_result = FrameOcrResult(
                frame_index=0,
                timestamp_ms=0,
                image_path=image_path,
                lines=[OcrLine("subtitle", 0.9, box=[10, 35, 70, 50])],
            )
            output_path = draw_debug_image(
                frame_result,
                output_dir=temp_path / "debug",
                frame_width=80,
                frame_height=60,
                min_confidence=0.5,
                subtitle_region={"enabled": True, "x_min": 0.0, "y_min": 0.5, "x_max": 1.0, "y_max": 1.0},
            )
            self.assertTrue(output_path.exists())

    def test_sample_video_frames(self) -> None:
        cv2 = __import__("cv2")
        import numpy as np

        from src.frame_sampler import sample_video_frames, sample_video_frames_between

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter is not available in this environment.")

            for index in range(8):
                frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            frames = sample_video_frames(
                video_path=video_path,
                output_dir=temp_path / "frames",
                interval_seconds=0.4,
                max_frames=3,
            )
            self.assertGreaterEqual(len(frames), 2)
            self.assertTrue(frames[0].image_path.exists())

            window_frames = sample_video_frames_between(
                video_path=video_path,
                output_dir=temp_path / "frames",
                start_ms=400,
                end_ms=1000,
                interval_ms=200,
                subdir_name="window",
            )
            self.assertGreaterEqual(len(window_frames), 3)
            self.assertEqual(window_frames[0].timestamp_ms, 400)
            self.assertTrue(window_frames[0].image_path.exists())

            memory_frames = sample_video_frames(
                video_path=video_path,
                output_dir=temp_path / "memory_frames",
                interval_seconds=0.4,
                max_frames=3,
                save_images=False,
                keep_images=True,
                decode_mode="sequential",
            )
            self.assertGreaterEqual(len(memory_frames), 2)
            self.assertFalse(memory_frames[0].image_path.exists())
            self.assertIsNotNone(memory_frames[0].image)
            self.assertEqual(memory_frames[0].width, 64)
            self.assertEqual(memory_frames[0].height, 48)

    def test_refine_segments_boundaries(self) -> None:
        cv2 = __import__("cv2")
        import numpy as np

        class FakeEngine:
            def predict_frame(self, sampled_frame: SampledFrame) -> FrameOcrResult:
                if 1250 <= sampled_frame.timestamp_ms < 2500:
                    lines = [OcrLine("hello", 0.9, box=[10, 10, 40, 25])]
                else:
                    lines = []
                return FrameOcrResult(
                    frame_index=sampled_frame.index,
                    timestamp_ms=sampled_frame.timestamp_ms,
                    image_path=sampled_frame.image_path,
                    lines=lines,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("OpenCV VideoWriter is not available in this environment.")

            for index in range(40):
                frame = np.full((48, 64, 3), index * 4, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            segments = [SubtitleSegment(start_ms=2000, end_ms=3000, text="hello", confidence=0.9)]
            coarse_results = [
                FrameOcrResult(
                    frame_index=0,
                    timestamp_ms=2000,
                    image_path=temp_path / "coarse.jpg",
                    lines=[OcrLine("hello", 0.9, box=[10, 10, 40, 25])],
                )
            ]
            refined, stats = refine_segments_boundaries(
                video_path=video_path,
                segments=segments,
                coarse_frame_results=coarse_results,
                engine=FakeEngine(),
                frames_dir=temp_path / "frames",
                coarse_interval_ms=1000,
                refine_window_ms=1000,
                refine_interval_ms=250,
                image_ext="jpg",
                min_confidence=0.5,
                subtitle_region={"enabled": False},
                noise_config={"enabled": False},
                sort_by_layout=True,
                row_tolerance=0.6,
                text_similarity_threshold=0.85,
                use_position=False,
                position_match_threshold=0.5,
                line_text_similarity_threshold=0.85,
                line_iou_threshold=0.3,
                line_center_distance_threshold=0.08,
            )
            self.assertEqual(refined[0].start_ms, 1250)
            self.assertEqual(refined[0].end_ms, 2500)
            self.assertEqual(stats.start_updates, 1)
            self.assertEqual(stats.end_updates, 1)
            self.assertGreater(stats.extra_ocr_frames, 0)


if __name__ == "__main__":
    unittest.main()
