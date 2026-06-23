from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.frame_sampler import sample_video_frames_at_timestamps
from src.models import FrameOcrResult, SampledFrame, SubtitleSegment
from src.subtitle_merger import line_match_score, text_similarity
from src.text_filter import filter_frame_result, frame_text
from src.utils import get_image_size


@dataclass
class BoundaryRefinementStats:
    enabled: bool
    refine_window_ms: int
    refine_interval_ms: int
    ocr_batch_size: int = 1
    segment_count: int = 0
    refined_segments: int = 0
    start_updates: int = 0
    end_updates: int = 0
    extra_sampled_frames: int = 0
    extra_ocr_frames: int = 0
    skipped_by_budget: int = 0
    max_extra_ocr_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "refine_window_ms": self.refine_window_ms,
            "refine_window_seconds": round(self.refine_window_ms / 1000, 4),
            "refine_interval_ms": self.refine_interval_ms,
            "refine_interval_seconds": round(self.refine_interval_ms / 1000, 4),
            "ocr_batch_size": self.ocr_batch_size,
            "segment_count": self.segment_count,
            "refined_segments": self.refined_segments,
            "start_updates": self.start_updates,
            "end_updates": self.end_updates,
            "extra_sampled_frames": self.extra_sampled_frames,
            "extra_ocr_frames": self.extra_ocr_frames,
            "skipped_by_budget": self.skipped_by_budget,
            "max_extra_ocr_frames": self.max_extra_ocr_frames,
        }


def find_reference_frame(
    segment: SubtitleSegment,
    frame_results: list[FrameOcrResult],
) -> FrameOcrResult | None:
    best_frame: FrameOcrResult | None = None
    best_score = 0.0
    for frame_result in frame_results:
        current_text = frame_text(frame_result.lines)
        line_scores = [
            text_similarity(segment.text, line.text)
            for line in frame_result.lines
            if line.text
        ]
        line_score = max(line_scores) if line_scores else 0.0
        if not current_text and line_score <= 0.0:
            continue

        score = max(text_similarity(segment.text, current_text), line_score)
        if segment.start_ms <= frame_result.timestamp_ms < segment.end_ms:
            score += 0.05
        if score > best_score:
            best_score = score
            best_frame = frame_result

    return best_frame


def frame_matches_reference(
    frame_result: FrameOcrResult,
    reference_frame: FrameOcrResult | None,
    reference_text: str,
    text_similarity_threshold: float,
    use_position: bool,
    position_match_threshold: float,
    line_text_similarity_threshold: float,
    line_iou_threshold: float,
    line_center_distance_threshold: float,
    frame_width: int,
    frame_height: int,
) -> bool:
    current_text = frame_text(frame_result.lines)
    whole_frame_match = text_similarity(reference_text, current_text) >= text_similarity_threshold
    current_reference_lines = [
        line
        for line in frame_result.lines
        if text_similarity(reference_text, line.text) >= text_similarity_threshold
    ]
    if not whole_frame_match and not current_reference_lines:
        return False

    if not use_position or reference_frame is None or not reference_frame.lines:
        return True

    reference_lines = [
        line
        for line in reference_frame.lines
        if text_similarity(reference_text, line.text) >= line_text_similarity_threshold
    ]
    if not reference_lines:
        reference_lines = reference_frame.lines
    candidate_lines = current_reference_lines or frame_result.lines
    position_score = line_match_score(
        reference_lines,
        candidate_lines,
        frame_width=frame_width,
        frame_height=frame_height,
        text_similarity_threshold=line_text_similarity_threshold,
        iou_threshold=line_iou_threshold,
        center_distance_threshold=line_center_distance_threshold,
    )
    return position_score >= position_match_threshold


def _predict_and_filter_frame(
    engine: Any,
    sampled_frame: SampledFrame,
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
) -> tuple[FrameOcrResult, int, int]:
    raw_result = engine.predict_frame(sampled_frame)
    width = raw_result.width or sampled_frame.width
    height = raw_result.height or sampled_frame.height
    if width is None or height is None:
        width, height = get_image_size(sampled_frame.image_path)
    filtered_result = filter_frame_result(
        raw_result,
        min_confidence=min_confidence,
        frame_width=width,
        frame_height=height,
        subtitle_region=subtitle_region,
        noise_config=noise_config,
        sort_by_layout=sort_by_layout,
        row_tolerance=row_tolerance,
    )
    return filtered_result, width, height


def _predict_and_filter_frames(
    engine: Any,
    sampled_frames: list[SampledFrame],
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
) -> dict[int, tuple[FrameOcrResult, int, int]]:
    if not sampled_frames:
        return {}

    if hasattr(engine, "predict_frames"):
        raw_results = engine.predict_frames(sampled_frames)
    else:
        raw_results = [engine.predict_frame(sampled_frame) for sampled_frame in sampled_frames]

    results: dict[int, tuple[FrameOcrResult, int, int]] = {}
    for sampled_frame, raw_result in zip(sampled_frames, raw_results):
        width = raw_result.width or sampled_frame.width
        height = raw_result.height or sampled_frame.height
        if width is None or height is None:
            width, height = get_image_size(sampled_frame.image_path)
        filtered_result = filter_frame_result(
            raw_result,
            min_confidence=min_confidence,
            frame_width=width,
            frame_height=height,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
        )
        results[sampled_frame.timestamp_ms] = (filtered_result, width, height)
        sampled_frame.image = None

    return results


def _empty_boundary_result(sampled_frame: SampledFrame) -> tuple[FrameOcrResult, int, int]:
    empty_result = FrameOcrResult(
        frame_index=sampled_frame.index,
        timestamp_ms=sampled_frame.timestamp_ms,
        image_path=sampled_frame.image_path,
        lines=[],
    )
    width = sampled_frame.width
    height = sampled_frame.height
    if width is None or height is None:
        width, height = get_image_size(sampled_frame.image_path)
    return empty_result, width, height


def _get_boundary_frame_result(
    sampled_frame: SampledFrame,
    batch_frames: list[SampledFrame],
    engine: Any,
    cache: dict[int, tuple[FrameOcrResult, int, int]],
    stats: BoundaryRefinementStats,
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
) -> tuple[FrameOcrResult, int, int]:
    cached = cache.get(sampled_frame.timestamp_ms)
    if cached is not None:
        return cached

    if stats.max_extra_ocr_frames is not None and stats.extra_ocr_frames >= stats.max_extra_ocr_frames:
        stats.skipped_by_budget += 1
        result = _empty_boundary_result(sampled_frame)
        cache[sampled_frame.timestamp_ms] = result
        return result

    candidate_frames = batch_frames if stats.ocr_batch_size > 1 else [sampled_frame]
    frames_to_predict = [
        frame
        for frame in candidate_frames
        if frame.timestamp_ms not in cache
    ]
    if stats.max_extra_ocr_frames is not None:
        remaining_budget = stats.max_extra_ocr_frames - stats.extra_ocr_frames
        frames_to_predict = frames_to_predict[:max(0, remaining_budget)]

    if not frames_to_predict:
        stats.skipped_by_budget += 1
        result = _empty_boundary_result(sampled_frame)
        cache[sampled_frame.timestamp_ms] = result
        return result

    predicted_results = _predict_and_filter_frames(
        engine=engine,
        sampled_frames=frames_to_predict,
        min_confidence=min_confidence,
        subtitle_region=subtitle_region,
        noise_config=noise_config,
        sort_by_layout=sort_by_layout,
        row_tolerance=row_tolerance,
    )
    cache.update(predicted_results)
    stats.extra_ocr_frames += len(predicted_results)
    return cache.get(sampled_frame.timestamp_ms, _empty_boundary_result(sampled_frame))


def _window_timestamps(
    start_ms: int,
    end_ms: int,
    refine_interval_ms: int,
) -> list[int]:
    if start_ms > end_ms:
        return []
    timestamps: list[int] = []
    timestamp_ms = int(start_ms)
    while timestamp_ms <= end_ms:
        timestamps.append(timestamp_ms)
        timestamp_ms += refine_interval_ms
    if timestamps and timestamps[-1] != end_ms:
        timestamps.append(end_ms)
    if not timestamps:
        timestamps.append(end_ms)
    return sorted(set(timestamps))


def _collect_boundary_timestamps(
    segments: list[SubtitleSegment],
    refine_window_ms: int,
    refine_interval_ms: int,
) -> dict[tuple[int, str], list[int]]:
    windows: dict[tuple[int, str], list[int]] = {}
    for index, segment in enumerate(segments):
        start_window_start = max(0, segment.start_ms - refine_window_ms)
        windows[(index, "start")] = _window_timestamps(
            start_window_start,
            segment.start_ms,
            refine_interval_ms,
        )

        end_window_start = max(segment.start_ms, segment.end_ms - refine_window_ms)
        windows[(index, "end")] = _window_timestamps(
            end_window_start,
            segment.end_ms,
            refine_interval_ms,
        )
    return windows


def _sample_unique_boundary_frames(
    video_path: str | Path,
    frames_dir: str | Path,
    windows: dict[tuple[int, str], list[int]],
    coarse_frame_results: list[FrameOcrResult],
    image_ext: str,
    stats: BoundaryRefinementStats,
    save_frame_images: bool,
    keep_frame_images: bool,
    decode_mode: str,
) -> dict[int, SampledFrame]:
    coarse_timestamps = {frame_result.timestamp_ms for frame_result in coarse_frame_results}
    candidate_timestamps = {
        timestamp_ms
        for timestamps in windows.values()
        for timestamp_ms in timestamps
        if timestamp_ms not in coarse_timestamps
    }
    sampled_frames = sample_video_frames_at_timestamps(
        video_path=video_path,
        output_dir=frames_dir,
        timestamps_ms=sorted(candidate_timestamps),
        image_ext=image_ext,
        subdir_name="boundary_refine",
        save_images=save_frame_images,
        keep_images=keep_frame_images,
        decode_mode=decode_mode,
    )
    stats.extra_sampled_frames += len(sampled_frames)
    return sampled_frames


def refine_segment_start(
    segment: SubtitleSegment,
    sampled_frames: list[SampledFrame],
    reference_frame: FrameOcrResult | None,
    engine: Any,
    cache: dict[int, tuple[FrameOcrResult, int, int]],
    stats: BoundaryRefinementStats,
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
    text_similarity_threshold: float,
    use_position: bool,
    position_match_threshold: float,
    line_text_similarity_threshold: float,
    line_iou_threshold: float,
    line_center_distance_threshold: float,
    ocr_batch_size: int,
) -> int:
    for position, sampled_frame in enumerate(sampled_frames):
        frame_result, width, height = _get_boundary_frame_result(
            sampled_frame=sampled_frame,
            batch_frames=sampled_frames[position:position + ocr_batch_size],
            engine=engine,
            cache=cache,
            stats=stats,
            min_confidence=min_confidence,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
        )
        if frame_matches_reference(
            frame_result=frame_result,
            reference_frame=reference_frame,
            reference_text=segment.text,
            text_similarity_threshold=text_similarity_threshold,
            use_position=use_position,
            position_match_threshold=position_match_threshold,
            line_text_similarity_threshold=line_text_similarity_threshold,
            line_iou_threshold=line_iou_threshold,
            line_center_distance_threshold=line_center_distance_threshold,
            frame_width=width,
            frame_height=height,
        ):
            return sampled_frame.timestamp_ms

    return segment.start_ms


def refine_segment_end(
    segment: SubtitleSegment,
    sampled_frames: list[SampledFrame],
    reference_frame: FrameOcrResult | None,
    engine: Any,
    cache: dict[int, tuple[FrameOcrResult, int, int]],
    stats: BoundaryRefinementStats,
    refine_interval_ms: int,
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
    text_similarity_threshold: float,
    use_position: bool,
    position_match_threshold: float,
    line_text_similarity_threshold: float,
    line_iou_threshold: float,
    line_center_distance_threshold: float,
    ocr_batch_size: int,
) -> int:
    last_match_ms: int | None = None
    seen_match = False
    for position, sampled_frame in enumerate(sampled_frames):
        frame_result, width, height = _get_boundary_frame_result(
            sampled_frame=sampled_frame,
            batch_frames=sampled_frames[position:position + ocr_batch_size],
            engine=engine,
            cache=cache,
            stats=stats,
            min_confidence=min_confidence,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
        )
        is_match = frame_matches_reference(
            frame_result=frame_result,
            reference_frame=reference_frame,
            reference_text=segment.text,
            text_similarity_threshold=text_similarity_threshold,
            use_position=use_position,
            position_match_threshold=position_match_threshold,
            line_text_similarity_threshold=line_text_similarity_threshold,
            line_iou_threshold=line_iou_threshold,
            line_center_distance_threshold=line_center_distance_threshold,
            frame_width=width,
            frame_height=height,
        )
        if is_match:
            last_match_ms = sampled_frame.timestamp_ms
            seen_match = True
            continue
        if seen_match:
            break

    if last_match_ms is None:
        return segment.end_ms

    refined_end_ms = min(segment.end_ms, last_match_ms + refine_interval_ms)
    return max(segment.start_ms + refine_interval_ms, refined_end_ms)


def refine_segments_boundaries(
    video_path: str | Path,
    segments: list[SubtitleSegment],
    coarse_frame_results: list[FrameOcrResult],
    engine: Any,
    frames_dir: str | Path,
    coarse_interval_ms: int,
    refine_window_ms: int,
    refine_interval_ms: int,
    image_ext: str,
    min_confidence: float,
    subtitle_region: dict[str, float] | None,
    noise_config: dict[str, object] | None,
    sort_by_layout: bool,
    row_tolerance: float,
    text_similarity_threshold: float,
    use_position: bool,
    position_match_threshold: float,
    line_text_similarity_threshold: float,
    line_iou_threshold: float,
    line_center_distance_threshold: float,
    max_extra_ocr_frames: int | None = None,
    save_frame_images: bool = False,
    keep_frame_images: bool = True,
    decode_mode: str = "sequential",
    ocr_batch_size: int = 1,
) -> tuple[list[SubtitleSegment], BoundaryRefinementStats]:
    refine_window_ms = max(refine_interval_ms, int(refine_window_ms))
    refine_window_ms = min(refine_window_ms, max(refine_interval_ms, int(coarse_interval_ms)))
    ocr_batch_size = max(1, int(ocr_batch_size))
    stats = BoundaryRefinementStats(
        enabled=True,
        refine_window_ms=refine_window_ms,
        refine_interval_ms=refine_interval_ms,
        ocr_batch_size=ocr_batch_size,
        segment_count=len(segments),
        max_extra_ocr_frames=max_extra_ocr_frames,
    )
    cache: dict[int, tuple[FrameOcrResult, int, int]] = {}
    sampled_frame_lookup: dict[int, SampledFrame] = {}
    for frame_result in coarse_frame_results:
        width = frame_result.width
        height = frame_result.height
        if width is None or height is None:
            if not frame_result.image_path.exists():
                continue
            try:
                width, height = get_image_size(frame_result.image_path)
            except ValueError:
                continue
        cache[frame_result.timestamp_ms] = (frame_result, width, height)
        sampled_frame_lookup[frame_result.timestamp_ms] = SampledFrame(
            index=frame_result.frame_index,
            timestamp_ms=frame_result.timestamp_ms,
            image_path=frame_result.image_path,
            width=width,
            height=height,
            image_saved=frame_result.image_path.exists(),
        )

    windows = _collect_boundary_timestamps(
        segments=segments,
        refine_window_ms=refine_window_ms,
        refine_interval_ms=refine_interval_ms,
    )
    sampled_frame_lookup.update(
        _sample_unique_boundary_frames(
            video_path=video_path,
            frames_dir=frames_dir,
            windows=windows,
            coarse_frame_results=coarse_frame_results,
            image_ext=image_ext,
            stats=stats,
            save_frame_images=save_frame_images,
            keep_frame_images=keep_frame_images,
            decode_mode=decode_mode,
        )
    )

    refined_segments: list[SubtitleSegment] = []

    for index, segment in enumerate(segments):
        reference_frame = find_reference_frame(segment, coarse_frame_results)
        start_frames = [
            sampled_frame_lookup[timestamp_ms]
            for timestamp_ms in windows.get((index, "start"), [])
            if timestamp_ms in sampled_frame_lookup
        ]
        end_frames = [
            sampled_frame_lookup[timestamp_ms]
            for timestamp_ms in windows.get((index, "end"), [])
            if timestamp_ms in sampled_frame_lookup
        ]
        refined_start_ms = refine_segment_start(
            segment=segment,
            sampled_frames=start_frames,
            reference_frame=reference_frame,
            engine=engine,
            cache=cache,
            stats=stats,
            min_confidence=min_confidence,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
            text_similarity_threshold=text_similarity_threshold,
            use_position=use_position,
            position_match_threshold=position_match_threshold,
            line_text_similarity_threshold=line_text_similarity_threshold,
            line_iou_threshold=line_iou_threshold,
            line_center_distance_threshold=line_center_distance_threshold,
            ocr_batch_size=ocr_batch_size,
        )
        refined_end_ms = refine_segment_end(
            segment=segment,
            sampled_frames=end_frames,
            reference_frame=reference_frame,
            engine=engine,
            cache=cache,
            stats=stats,
            refine_interval_ms=refine_interval_ms,
            min_confidence=min_confidence,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
            text_similarity_threshold=text_similarity_threshold,
            use_position=use_position,
            position_match_threshold=position_match_threshold,
            line_text_similarity_threshold=line_text_similarity_threshold,
            line_iou_threshold=line_iou_threshold,
            line_center_distance_threshold=line_center_distance_threshold,
            ocr_batch_size=ocr_batch_size,
        )

        if refined_start_ms != segment.start_ms:
            stats.start_updates += 1
        if refined_end_ms != segment.end_ms:
            stats.end_updates += 1
        if refined_start_ms != segment.start_ms or refined_end_ms != segment.end_ms:
            stats.refined_segments += 1

        if refined_end_ms <= refined_start_ms:
            refined_end_ms = max(segment.end_ms, refined_start_ms + refine_interval_ms)

        refined_segments.append(
            SubtitleSegment(
                start_ms=refined_start_ms,
                end_ms=refined_end_ms,
                text=segment.text,
                confidence=segment.confidence,
            )
        )

    return refined_segments, stats
