from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from src.batch_runner import (
    BatchVideoResult,
    build_video_frames_dir,
    build_video_output_dir,
    copy_srt_to_video_dir,
    discover_video_files,
    export_batch_summary_json,
    export_batch_summary_txt,
    find_srt_output_path,
    parse_video_extensions,
)
from src.boundary_refiner import refine_segments_boundaries
from src.debug_visualizer import draw_debug_image
from src.exporters import (
    export_frame_results_json,
    export_sampled_frames_json,
    export_segments_json,
    export_srt,
    export_txt,
)
from src.frame_sampler import sample_video_frames
from src.models import FrameOcrResult
from src.ocr_cache import (
    OcrReuseStats,
    clone_ocr_result_for_frame,
    frame_difference_metrics,
    make_frame_fingerprint,
)
from src.ocr_engine import PaddleOCREngine
from src.run_report import build_run_report, export_run_report_json, export_run_report_txt
from src.subtitle_merger import merge_repeated_text_snapshots, merge_text_tracks_state_machine
from src.text_filter import filter_frame_result
from src.utils import get_config_value, get_image_size, load_config
from src.video_reader import read_video_info


def optional_int(config: dict, key: str) -> int | None:
    value = get_config_value(config, key)
    return None if value is None else int(value)


def optional_float(config: dict, key: str) -> float | None:
    value = get_config_value(config, key)
    return None if value is None else float(value)


def optional_str(config: dict, key: str) -> str | None:
    value = get_config_value(config, key)
    return None if value is None else str(value)


def optional_bool(config: dict, key: str) -> bool | None:
    value = get_config_value(config, key)
    return None if value is None else bool(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract video subtitles and on-screen text.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--video-dir", default=None, help="Directory containing videos for batch OCR.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --video-dir.")
    parser.add_argument(
        "--video-extensions",
        default=None,
        help="Comma-separated video extensions for batch mode, for example mp4,mov,mkv.",
    )
    parser.add_argument("--frames-dir", default=None, help="Directory to save sampled frames.")
    parser.add_argument("--output-dir", default=None, help="Directory to save outputs.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames for quick tests.")
    parser.add_argument("--device", default=None, help="PaddleOCR device, for example cpu or gpu:0.")
    parser.add_argument("--sample-only", action="store_true", help="Only sample frames, skip OCR.")
    parser.set_defaults(copy_srt_to_video_dir=None)
    parser.add_argument(
        "--copy-srt-to-video-dir",
        dest="copy_srt_to_video_dir",
        action="store_true",
        help="Copy generated SRT files back to each source video directory in batch mode.",
    )
    parser.add_argument(
        "--no-copy-srt-to-video-dir",
        dest="copy_srt_to_video_dir",
        action="store_false",
        help="Do not copy generated SRT files back to each source video directory in batch mode.",
    )
    return parser


def run_pipeline(args: argparse.Namespace, engine: PaddleOCREngine | None = None) -> list[Path]:
    pipeline_started_at = perf_counter()
    timings: dict[str, float] = {
        "video_info_seconds": 0.0,
        "sampling_seconds": 0.0,
        "ocr_filter_debug_seconds": 0.0,
        "merge_seconds": 0.0,
        "boundary_refinement_seconds": 0.0,
        "export_seconds": 0.0,
        "total_seconds": 0.0,
    }

    config = load_config(args.config)
    video_path = args.video or get_config_value(config, "input.video_path")
    if not video_path:
        raise ValueError("Please provide --video or set input.video_path in config.")

    frames_dir = Path(args.frames_dir or get_config_value(config, "output.frames_dir", "data/frames"))
    output_dir = Path(args.output_dir or get_config_value(config, "output.output_dir", "data/outputs"))
    sampling_strategy = str(get_config_value(config, "sampling.strategy", "fixed")).lower()
    interval_seconds = float(get_config_value(config, "sampling.interval_seconds", 1.0))
    coarse_interval_seconds = float(
        get_config_value(config, "sampling.coarse_interval_seconds", interval_seconds)
    )
    sample_interval_seconds = (
        coarse_interval_seconds if sampling_strategy == "adaptive_boundary" else interval_seconds
    )
    if sampling_strategy not in {"fixed", "adaptive_boundary"}:
        raise ValueError(f"Unsupported sampling.strategy: {sampling_strategy}")

    refine_window_seconds = float(get_config_value(config, "sampling.refine_window_seconds", 1.0))
    refine_interval_seconds = float(get_config_value(config, "sampling.refine_interval_seconds", 0.1))
    refine_ocr_batch_size = max(1, int(get_config_value(config, "sampling.refine_ocr_batch_size", 1)))
    refine_max_extra_ocr_frames = get_config_value(config, "sampling.refine_max_extra_ocr_frames")
    if refine_max_extra_ocr_frames is not None:
        refine_max_extra_ocr_frames = int(refine_max_extra_ocr_frames)
    image_ext = str(get_config_value(config, "sampling.image_ext", "jpg"))
    decode_mode = str(get_config_value(config, "sampling.decode_mode", "sequential")).lower()
    max_frames = args.max_frames
    if max_frames is None:
        max_frames = get_config_value(config, "sampling.max_frames")

    formats = get_config_value(config, "output.formats", ["json", "txt", "srt"])
    report_enabled = bool(get_config_value(config, "output.report.enabled", True))
    report_formats = get_config_value(config, "output.report.formats", ["json", "txt"])

    ocr_device = args.device or get_config_value(config, "ocr.device", "cpu")
    ocr_batch_size = max(1, int(get_config_value(config, "ocr.batch_size", 1)))
    min_confidence = float(get_config_value(config, "filter.min_confidence", 0.0))
    noise_config = get_config_value(config, "filter.noise", {"enabled": False})
    subtitle_region = get_config_value(
        config,
        "filter.text_region",
        get_config_value(config, "filter.subtitle_region", {}),
    )
    sort_lines = bool(get_config_value(config, "layout.sort_lines", True))
    row_tolerance = float(get_config_value(config, "layout.row_tolerance", 0.6))
    debug_enabled = bool(get_config_value(config, "debug.enabled", False))
    debug_output_dir_name = str(get_config_value(config, "debug.output_dir_name", "debug"))
    save_frame_images = bool(get_config_value(config, "sampling.save_frame_images", False))
    keep_frame_images = bool(get_config_value(config, "sampling.keep_frame_images_in_memory", True))
    actual_save_frame_images = save_frame_images or debug_enabled or bool(args.sample_only)
    actual_keep_frame_images = keep_frame_images and not bool(args.sample_only)
    ocr_cache_requested = bool(get_config_value(config, "ocr_cache.enabled", False))
    ocr_cache_threshold = float(get_config_value(config, "ocr_cache.image_diff_threshold", 0.5))
    ocr_cache_max_pixel_threshold = float(
        get_config_value(config, "ocr_cache.max_pixel_diff_threshold", 12.0)
    )
    ocr_cache_image_size = int(get_config_value(config, "ocr_cache.image_size", 96))
    ocr_cache_disabled_reason = None
    actual_ocr_cache_enabled = ocr_cache_requested
    if actual_ocr_cache_enabled and debug_enabled:
        actual_ocr_cache_enabled = False
        ocr_cache_disabled_reason = "debug_enabled"
    if actual_ocr_cache_enabled and ocr_batch_size > 1:
        actual_ocr_cache_enabled = False
        ocr_cache_disabled_reason = "batch_ocr_enabled"
    if actual_ocr_cache_enabled and not actual_keep_frame_images:
        actual_ocr_cache_enabled = False
        ocr_cache_disabled_reason = "in_memory_frames_disabled"
    ocr_reuse_stats = OcrReuseStats(
        enabled=actual_ocr_cache_enabled,
        disabled_reason=ocr_cache_disabled_reason,
        image_diff_threshold=ocr_cache_threshold,
        max_pixel_diff_threshold=ocr_cache_max_pixel_threshold,
        image_size=ocr_cache_image_size,
    )
    merge_settings = {
        "strategy": str(get_config_value(config, "merge.strategy", "snapshot")).lower(),
        "similarity_threshold": float(get_config_value(config, "merge.similarity_threshold", 0.92)),
        "max_gap_seconds": float(get_config_value(config, "merge.max_gap_seconds", 1.5)),
        "use_position": bool(get_config_value(config, "merge.use_position", True)),
        "position_match_threshold": float(
            get_config_value(config, "merge.position_match_threshold", 0.5)
        ),
        "line_text_similarity_threshold": float(
            get_config_value(config, "merge.line_text_similarity_threshold", 0.85)
        ),
        "line_iou_threshold": float(get_config_value(config, "merge.line_iou_threshold", 0.3)),
        "line_center_distance_threshold": float(
            get_config_value(config, "merge.line_center_distance_threshold", 0.08)
        ),
    }
    boundary_text_similarity_threshold = float(
        get_config_value(
            config,
            "sampling.boundary_text_similarity_threshold",
            max(0.8, merge_settings["similarity_threshold"] - 0.07),
        )
    )

    info_started_at = perf_counter()
    video_info = read_video_info(video_path)
    timings["video_info_seconds"] = perf_counter() - info_started_at

    sampling_started_at = perf_counter()
    sampled_frames = sample_video_frames(
        video_path=video_path,
        output_dir=frames_dir,
        interval_seconds=sample_interval_seconds,
        max_frames=max_frames,
        image_ext=image_ext,
        save_images=actual_save_frame_images,
        keep_images=actual_keep_frame_images,
        decode_mode=decode_mode,
    )
    timings["sampling_seconds"] = perf_counter() - sampling_started_at

    video_stem = Path(video_path).stem
    written_paths: list[Path] = []
    sampled_index_path = output_dir / f"{video_stem}_sampled_frames.json"
    export_started_at = perf_counter()
    written_paths.append(export_sampled_frames_json(sampled_index_path, sampled_frames))
    timings["export_seconds"] += perf_counter() - export_started_at
    boundary_refinement_summary = {
        "enabled": sampling_strategy == "adaptive_boundary",
        "refine_window_ms": int(refine_window_seconds * 1000),
        "refine_window_seconds": refine_window_seconds,
        "refine_interval_ms": int(refine_interval_seconds * 1000),
        "refine_interval_seconds": refine_interval_seconds,
        "ocr_batch_size": refine_ocr_batch_size,
        "segment_count": 0,
        "refined_segments": 0,
        "start_updates": 0,
        "end_updates": 0,
        "extra_sampled_frames": 0,
        "extra_ocr_frames": 0,
        "skipped_by_budget": 0,
        "max_extra_ocr_frames": refine_max_extra_ocr_frames,
    }

    def write_report(
        frame_results: list[FrameOcrResult],
        segments: list,
    ) -> list[Path]:
        if not report_enabled:
            return []

        report_paths: list[Path] = []
        if "json" in report_formats:
            report_paths.append(output_dir / f"{video_stem}_report.json")
        if "txt" in report_formats:
            report_paths.append(output_dir / f"{video_stem}_report.txt")

        timings["total_seconds"] = perf_counter() - pipeline_started_at
        settings = {
            "command": " ".join(sys.argv),
            "config_path": str(Path(args.config)),
            "sample_only": bool(args.sample_only),
            "sampling": {
                "strategy": sampling_strategy,
                "interval_seconds": interval_seconds,
                "coarse_interval_seconds": coarse_interval_seconds,
                "actual_interval_seconds": sample_interval_seconds,
                "refine_window_seconds": refine_window_seconds,
                "refine_interval_seconds": refine_interval_seconds,
                "refine_ocr_batch_size": refine_ocr_batch_size,
                "refine_max_extra_ocr_frames": refine_max_extra_ocr_frames,
                "boundary_text_similarity_threshold": boundary_text_similarity_threshold,
                "max_frames": max_frames,
                "image_ext": image_ext,
                "decode_mode": decode_mode,
                "save_frame_images": actual_save_frame_images,
                "keep_frame_images_in_memory": actual_keep_frame_images,
            },
            "ocr": {
                "engine": get_config_value(config, "ocr.engine", "paddleocr"),
                "device": ocr_device,
                "batch_size": ocr_batch_size,
                "text_detection_model_name": get_config_value(
                    config, "ocr.text_detection_model_name", "PP-OCRv6_medium_det"
                ),
                "text_recognition_model_name": get_config_value(
                    config, "ocr.text_recognition_model_name", "PP-OCRv6_medium_rec"
                ),
                "text_score_threshold": float(
                    get_config_value(config, "ocr.text_score_threshold", 0.0)
                ),
                "text_det_limit_side_len": get_config_value(config, "ocr.text_det_limit_side_len"),
                "text_det_limit_type": get_config_value(config, "ocr.text_det_limit_type"),
                "text_det_thresh": get_config_value(config, "ocr.text_det_thresh"),
                "text_det_box_thresh": get_config_value(config, "ocr.text_det_box_thresh"),
                "text_det_unclip_ratio": get_config_value(config, "ocr.text_det_unclip_ratio"),
                "text_rec_score_thresh": get_config_value(config, "ocr.text_rec_score_thresh"),
                "return_word_box": get_config_value(config, "ocr.return_word_box"),
            },
            "ocr_cache": ocr_reuse_stats.to_dict(),
            "filter": {
                "min_confidence": min_confidence,
                "noise": noise_config,
                "text_region": subtitle_region,
            },
            "layout": {
                "sort_lines": sort_lines,
                "row_tolerance": row_tolerance,
            },
            "debug": {
                "enabled": debug_enabled,
                "output_dir_name": debug_output_dir_name,
            },
            "merge": merge_settings,
            "boundary_refinement": boundary_refinement_summary,
            "output": {
                "formats": formats,
                "report_formats": report_formats,
            },
        }
        report = build_run_report(
            video_info=video_info,
            sampled_frames=sampled_frames,
            frame_results=frame_results,
            segments=segments,
            written_paths=[*written_paths, *report_paths],
            timings=timings,
            settings=settings,
        )

        written_report_paths: list[Path] = []
        report_export_started_at = perf_counter()
        for report_path in report_paths:
            if report_path.suffix == ".json":
                written_report_paths.append(export_run_report_json(report_path, report))
            if report_path.suffix == ".txt":
                written_report_paths.append(export_run_report_txt(report_path, report))
        timings["export_seconds"] += perf_counter() - report_export_started_at
        return written_report_paths

    if args.sample_only:
        written_paths.extend(write_report(frame_results=[], segments=[]))
        return written_paths

    if engine is None:
        engine = build_shared_engine(args, config)

    frame_results: list[FrameOcrResult] = []
    debug_output_dir = output_dir / debug_output_dir_name / video_stem
    debug_written = False
    previous_fingerprint = None
    previous_filtered_result: FrameOcrResult | None = None

    def filter_and_maybe_debug(
        raw_result: FrameOcrResult,
        sampled_frame,
    ) -> FrameOcrResult:
        nonlocal debug_written

        width = raw_result.width or sampled_frame.width
        height = raw_result.height or sampled_frame.height
        if width is None or height is None:
            width, height = get_image_size(sampled_frame.image_path)
        if debug_enabled:
            draw_debug_image(
                raw_result,
                output_dir=debug_output_dir,
                frame_width=width,
                frame_height=height,
                min_confidence=min_confidence,
                subtitle_region=subtitle_region,
                noise_config=noise_config,
                draw_rejected=bool(get_config_value(config, "debug.draw_rejected", True)),
                line_thickness=int(get_config_value(config, "debug.line_thickness", 2)),
            )
            debug_written = True
        return filter_frame_result(
            raw_result,
            min_confidence=min_confidence,
            frame_width=width,
            frame_height=height,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_lines,
            row_tolerance=row_tolerance,
        )

    ocr_started_at = perf_counter()
    if ocr_batch_size > 1 and not actual_ocr_cache_enabled:
        for start_index in range(0, len(sampled_frames), ocr_batch_size):
            batch_frames = sampled_frames[start_index:start_index + ocr_batch_size]
            raw_results = engine.predict_frames(batch_frames)
            ocr_reuse_stats.ocr_frames += len(batch_frames)
            for sampled_frame, raw_result in zip(batch_frames, raw_results):
                frame_results.append(filter_and_maybe_debug(raw_result, sampled_frame))
                sampled_frame.image = None
    else:
        for sampled_frame in sampled_frames:
            current_fingerprint = None
            if actual_ocr_cache_enabled:
                if sampled_frame.image is None:
                    ocr_reuse_stats.skipped_no_image += 1
                else:
                    current_fingerprint = make_frame_fingerprint(
                        sampled_frame.image,
                        image_size=ocr_cache_image_size,
                    )
                    if previous_fingerprint is None or previous_filtered_result is None:
                        ocr_reuse_stats.skipped_no_reference += 1
                    else:
                        metrics = frame_difference_metrics(previous_fingerprint, current_fingerprint)
                        ocr_reuse_stats.record_difference(metrics)
                        if metrics.is_reusable(
                            mean_threshold=ocr_cache_threshold,
                            max_threshold=ocr_cache_max_pixel_threshold,
                        ):
                            reused_result = clone_ocr_result_for_frame(
                                previous_filtered_result,
                                sampled_frame,
                            )
                            frame_results.append(reused_result)
                            ocr_reuse_stats.reused_frames += 1
                            previous_fingerprint = current_fingerprint
                            previous_filtered_result = reused_result
                            sampled_frame.image = None
                            continue

            raw_result = engine.predict_frame(sampled_frame)
            ocr_reuse_stats.ocr_frames += 1
            filtered_result = filter_and_maybe_debug(raw_result, sampled_frame)
            frame_results.append(filtered_result)
            sampled_frame.image = None
            if actual_ocr_cache_enabled:
                previous_fingerprint = current_fingerprint
                previous_filtered_result = filtered_result if current_fingerprint is not None else None
    timings["ocr_filter_debug_seconds"] = perf_counter() - ocr_started_at

    frame_results_path = output_dir / f"{video_stem}_frame_ocr.json"
    export_started_at = perf_counter()
    written_paths.append(export_frame_results_json(frame_results_path, frame_results))
    if debug_written:
        written_paths.append(debug_output_dir)
    timings["export_seconds"] += perf_counter() - export_started_at

    frame_interval_ms = int(sample_interval_seconds * 1000)
    max_gap_ms = int(merge_settings["max_gap_seconds"] * 1000)
    merge_started_at = perf_counter()
    if merge_settings["strategy"] == "line_state_machine":
        segments = merge_text_tracks_state_machine(
            frame_results,
            frame_interval_ms=frame_interval_ms,
            max_gap_ms=max_gap_ms,
            use_position=merge_settings["use_position"],
            line_text_similarity_threshold=merge_settings["line_text_similarity_threshold"],
            line_iou_threshold=merge_settings["line_iou_threshold"],
            line_center_distance_threshold=merge_settings["line_center_distance_threshold"],
        )
    elif merge_settings["strategy"] == "snapshot":
        segments = merge_repeated_text_snapshots(
            frame_results,
            frame_interval_ms=frame_interval_ms,
            similarity_threshold=merge_settings["similarity_threshold"],
            max_gap_ms=max_gap_ms,
            use_position=merge_settings["use_position"],
            position_match_threshold=merge_settings["position_match_threshold"],
            line_text_similarity_threshold=merge_settings["line_text_similarity_threshold"],
            line_iou_threshold=merge_settings["line_iou_threshold"],
            line_center_distance_threshold=merge_settings["line_center_distance_threshold"],
        )
    else:
        raise ValueError(f"Unsupported merge.strategy: {merge_settings['strategy']}")
    timings["merge_seconds"] = perf_counter() - merge_started_at

    if sampling_strategy == "adaptive_boundary":
        refine_started_at = perf_counter()
        refine_interval_ms = max(1, int(refine_interval_seconds * 1000))
        segments, refine_stats = refine_segments_boundaries(
            video_path=video_path,
            segments=segments,
            coarse_frame_results=frame_results,
            engine=engine,
            frames_dir=frames_dir,
            coarse_interval_ms=frame_interval_ms,
            refine_window_ms=int(refine_window_seconds * 1000),
            refine_interval_ms=refine_interval_ms,
            image_ext=image_ext,
            min_confidence=min_confidence,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_lines,
            row_tolerance=row_tolerance,
            text_similarity_threshold=boundary_text_similarity_threshold,
            use_position=merge_settings["use_position"],
            position_match_threshold=merge_settings["position_match_threshold"],
            line_text_similarity_threshold=merge_settings["line_text_similarity_threshold"],
            line_iou_threshold=merge_settings["line_iou_threshold"],
            line_center_distance_threshold=merge_settings["line_center_distance_threshold"],
            max_extra_ocr_frames=refine_max_extra_ocr_frames,
            save_frame_images=save_frame_images or debug_enabled,
            keep_frame_images=keep_frame_images,
            decode_mode=decode_mode,
            ocr_batch_size=refine_ocr_batch_size,
        )
        boundary_refinement_summary = refine_stats.to_dict()
        timings["boundary_refinement_seconds"] = perf_counter() - refine_started_at

    export_started_at = perf_counter()
    if "json" in formats:
        written_paths.append(export_segments_json(output_dir / f"{video_stem}_segments.json", segments))
    if "txt" in formats:
        written_paths.append(export_txt(output_dir / f"{video_stem}.txt", segments))
    if "srt" in formats:
        written_paths.append(export_srt(output_dir / f"{video_stem}.srt", segments))
    timings["export_seconds"] += perf_counter() - export_started_at

    written_paths.extend(write_report(frame_results=frame_results, segments=segments))

    return written_paths


def build_shared_engine(args: argparse.Namespace, config: dict) -> PaddleOCREngine:
    return PaddleOCREngine(
        text_detection_model_name=get_config_value(
            config, "ocr.text_detection_model_name", "PP-OCRv6_medium_det"
        ),
        text_recognition_model_name=get_config_value(
            config, "ocr.text_recognition_model_name", "PP-OCRv6_medium_rec"
        ),
        device=args.device or get_config_value(config, "ocr.device", "cpu"),
        use_doc_orientation_classify=bool(
            get_config_value(config, "ocr.use_doc_orientation_classify", False)
        ),
        use_doc_unwarping=bool(get_config_value(config, "ocr.use_doc_unwarping", False)),
        use_textline_orientation=bool(get_config_value(config, "ocr.use_textline_orientation", True)),
        text_score_threshold=float(get_config_value(config, "ocr.text_score_threshold", 0.0)),
        text_det_limit_side_len=optional_int(config, "ocr.text_det_limit_side_len"),
        text_det_limit_type=optional_str(config, "ocr.text_det_limit_type"),
        text_det_thresh=optional_float(config, "ocr.text_det_thresh"),
        text_det_box_thresh=optional_float(config, "ocr.text_det_box_thresh"),
        text_det_unclip_ratio=optional_float(config, "ocr.text_det_unclip_ratio"),
        text_rec_score_thresh=optional_float(config, "ocr.text_rec_score_thresh"),
        return_word_box=optional_bool(config, "ocr.return_word_box"),
    )


def run_batch_pipeline(args: argparse.Namespace) -> list[Path]:
    if args.video:
        raise ValueError("Please use either --video or --video-dir, not both.")
    if not args.video_dir:
        raise ValueError("Please provide --video-dir for batch mode.")

    batch_started_at = perf_counter()
    config = load_config(args.config)
    video_dir = Path(args.video_dir)
    output_root = Path(args.output_dir or get_config_value(config, "output.output_dir", "data/outputs"))
    frames_root = Path(args.frames_dir) if args.frames_dir else None
    recursive = bool(args.recursive or get_config_value(config, "batch.recursive", False))
    extensions = parse_video_extensions(
        args.video_extensions or get_config_value(config, "batch.video_extensions")
    )
    copy_srt_enabled = args.copy_srt_to_video_dir
    if copy_srt_enabled is None:
        copy_srt_enabled = bool(get_config_value(config, "batch.copy_srt_to_video_dir", True))

    videos = discover_video_files(
        video_dir=video_dir,
        extensions=extensions,
        recursive=recursive,
    )
    if not videos:
        raise ValueError(f"No video files found in {video_dir} with extensions: {', '.join(extensions)}")

    shared_engine = None if args.sample_only else build_shared_engine(args, config)
    results: list[BatchVideoResult] = []
    summary_paths: list[Path] = []

    for index, video_path in enumerate(videos, start=1):
        video_started_at = perf_counter()
        video_output_dir = build_video_output_dir(
            video_path=video_path,
            video_root=video_dir,
            output_root=output_root,
        )
        video_frames_dir = build_video_frames_dir(
            video_path=video_path,
            video_root=video_dir,
            frames_root=frames_root,
            video_output_dir=video_output_dir,
        )
        print(f"[{index}/{len(videos)}] Processing: {video_path}")
        try:
            pipeline_args = argparse.Namespace(
                config=args.config,
                video=str(video_path),
                frames_dir=str(video_frames_dir),
                output_dir=str(video_output_dir),
                max_frames=args.max_frames,
                device=args.device,
                sample_only=args.sample_only,
            )
            written_paths = run_pipeline(pipeline_args, engine=shared_engine)
            copied_srt_path = None
            if copy_srt_enabled and not args.sample_only:
                srt_path = find_srt_output_path(written_paths, video_path.stem)
                if srt_path is not None:
                    copied_srt_path = copy_srt_to_video_dir(srt_path, video_path)
            results.append(
                BatchVideoResult(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    frames_dir=video_frames_dir,
                    status="success",
                    written_paths=written_paths,
                    copied_srt_path=copied_srt_path,
                    elapsed_seconds=perf_counter() - video_started_at,
                )
            )
        except Exception as exc:
            results.append(
                BatchVideoResult(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    frames_dir=video_frames_dir,
                    status="failed",
                    written_paths=[],
                    error=str(exc),
                    elapsed_seconds=perf_counter() - video_started_at,
                )
            )
            print(f"Failed: {video_path}\n  {exc}")

    total_elapsed_seconds = perf_counter() - batch_started_at
    summary_paths.append(
        export_batch_summary_json(
            output_root / "batch_summary.json",
            results=results,
            total_elapsed_seconds=total_elapsed_seconds,
        )
    )
    summary_paths.append(
        export_batch_summary_txt(
            output_root / "batch_summary.txt",
            results=results,
            total_elapsed_seconds=total_elapsed_seconds,
        )
    )
    return summary_paths


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    written_paths = run_batch_pipeline(args) if args.video_dir else run_pipeline(args)
    print("Done. Written files:")
    for path in written_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
