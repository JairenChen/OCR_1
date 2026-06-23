from __future__ import annotations

from pathlib import Path

import cv2

from src.models import SampledFrame
from src.utils import ensure_dir
from src.video_reader import read_video_info


def _normalize_decode_mode(decode_mode: str) -> str:
    normalized = decode_mode.lower().strip()
    if normalized not in {"auto", "seek", "sequential"}:
        raise ValueError(f"Unsupported decode_mode: {decode_mode}")
    return normalized


def _resolve_decode_mode(
    decode_mode: str,
    timestamp_count: int,
    frame_count: int,
) -> str:
    normalized = _normalize_decode_mode(decode_mode)
    if normalized != "auto":
        return normalized
    if frame_count <= 0:
        return "seek"

    sample_density = timestamp_count / frame_count
    return "sequential" if sample_density >= 0.15 else "seek"


def _normalize_timestamps(timestamps_ms: list[int], duration_ms: int) -> list[int]:
    if duration_ms > 0:
        return sorted(
            {
                max(0, min(int(timestamp_ms), duration_ms))
                for timestamp_ms in timestamps_ms
            }
        )
    return sorted({max(0, int(timestamp_ms)) for timestamp_ms in timestamps_ms})


def _make_sampled_frame(
    index: int,
    timestamp_ms: int,
    frame,
    frame_dir: Path,
    image_ext: str,
    save_images: bool,
    keep_images: bool,
) -> SampledFrame:
    image_path = frame_dir / f"frame_{index:06d}_{timestamp_ms:09d}ms.{image_ext}"
    if save_images and not cv2.imwrite(str(image_path), frame):
        raise OSError(f"Could not save frame image: {image_path}")

    height, width = frame.shape[:2]
    return SampledFrame(
        index=index,
        timestamp_ms=timestamp_ms,
        image_path=image_path,
        image=frame.copy() if keep_images else None,
        width=width,
        height=height,
        image_saved=save_images,
    )


def _current_frame_time_ms(capture, fps: float) -> int:
    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
    if position_ms > 0:
        return int(round(position_ms))

    frame_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 1.0) - 1.0
    if fps > 0:
        return int(round(frame_position / fps * 1000))
    return int(max(0, round(frame_position)))


def _read_frames_by_seeking(
    capture,
    timestamps: list[int],
    frame_dir: Path,
    image_ext: str,
    save_images: bool,
    keep_images: bool,
) -> dict[int, SampledFrame]:
    sampled_frames: dict[int, SampledFrame] = {}
    for timestamp_ms in timestamps:
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = capture.read()
        if not ok:
            continue

        sampled_frames[timestamp_ms] = _make_sampled_frame(
            index=len(sampled_frames),
            timestamp_ms=timestamp_ms,
            frame=frame,
            frame_dir=frame_dir,
            image_ext=image_ext,
            save_images=save_images,
            keep_images=keep_images,
        )
    return sampled_frames


def _read_frames_sequentially(
    capture,
    timestamps: list[int],
    frame_dir: Path,
    image_ext: str,
    save_images: bool,
    keep_images: bool,
    fps: float,
) -> dict[int, SampledFrame]:
    sampled_frames: dict[int, SampledFrame] = {}
    target_index = 0
    previous_time_ms: int | None = None
    previous_frame = None

    while target_index < len(timestamps):
        ok, current_frame = capture.read()
        if not ok:
            break

        current_time_ms = _current_frame_time_ms(capture, fps=fps)
        while target_index < len(timestamps) and timestamps[target_index] <= current_time_ms:
            target_ms = timestamps[target_index]
            selected_frame = current_frame
            if previous_frame is not None and previous_time_ms is not None:
                previous_distance = abs(previous_time_ms - target_ms)
                current_distance = abs(current_time_ms - target_ms)
                if previous_distance <= current_distance:
                    selected_frame = previous_frame

            sampled_frames[target_ms] = _make_sampled_frame(
                index=len(sampled_frames),
                timestamp_ms=target_ms,
                frame=selected_frame,
                frame_dir=frame_dir,
                image_ext=image_ext,
                save_images=save_images,
                keep_images=keep_images,
            )
            target_index += 1

        previous_time_ms = current_time_ms
        previous_frame = current_frame.copy()

    return sampled_frames


def sample_video_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval_seconds: float,
    max_frames: int | None = None,
    image_ext: str = "jpg",
    save_images: bool = True,
    keep_images: bool = False,
    decode_mode: str = "seek",
) -> list[SampledFrame]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than 0")

    video_info = read_video_info(video_path)
    step_ms = max(1, int(interval_seconds * 1000))
    timestamp_ms = 0
    timestamps: list[int] = []
    while True:
        if max_frames is not None and len(timestamps) >= max_frames:
            break
        if video_info.duration_ms > 0 and timestamp_ms > video_info.duration_ms:
            break
        timestamps.append(timestamp_ms)
        timestamp_ms += step_ms

    sampled_frames = sample_video_frames_at_timestamps(
        video_path=video_path,
        output_dir=output_dir,
        timestamps_ms=timestamps,
        image_ext=image_ext,
        save_images=save_images,
        keep_images=keep_images,
        decode_mode=decode_mode,
    )
    return [sampled_frames[timestamp] for timestamp in sorted(sampled_frames)]


def sample_video_frames_between(
    video_path: str | Path,
    output_dir: str | Path,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    image_ext: str = "jpg",
    subdir_name: str | None = None,
    save_images: bool = True,
    keep_images: bool = False,
    decode_mode: str = "seek",
) -> list[SampledFrame]:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be greater than 0")

    video_info = read_video_info(video_path)
    if video_info.duration_ms > 0:
        start_ms = max(0, min(int(start_ms), video_info.duration_ms))
        end_ms = max(0, min(int(end_ms), video_info.duration_ms))
    else:
        start_ms = max(0, int(start_ms))
        end_ms = max(0, int(end_ms))

    if start_ms > end_ms:
        return []

    timestamps: list[int] = []
    timestamp_ms = start_ms
    while timestamp_ms <= end_ms:
        timestamps.append(timestamp_ms)
        timestamp_ms += interval_ms

    sampled_frames = sample_video_frames_at_timestamps(
        video_path=video_path,
        output_dir=output_dir,
        timestamps_ms=timestamps,
        image_ext=image_ext,
        subdir_name=subdir_name,
        save_images=save_images,
        keep_images=keep_images,
        decode_mode=decode_mode,
    )
    return [sampled_frames[timestamp] for timestamp in sorted(sampled_frames)]


def sample_video_frames_at_timestamps(
    video_path: str | Path,
    output_dir: str | Path,
    timestamps_ms: list[int],
    image_ext: str = "jpg",
    subdir_name: str | None = None,
    save_images: bool = True,
    keep_images: bool = False,
    decode_mode: str = "seek",
) -> dict[int, SampledFrame]:
    video_info = read_video_info(video_path)
    timestamps = _normalize_timestamps(timestamps_ms, duration_ms=video_info.duration_ms)
    if not timestamps:
        return {}

    capture = cv2.VideoCapture(str(video_info.path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_info.path}")

    frame_dir = Path(output_dir) / video_info.path.stem
    if subdir_name:
        frame_dir = frame_dir / subdir_name
    if save_images:
        ensure_dir(frame_dir)

    image_ext = image_ext.lstrip(".")
    decode_mode = _resolve_decode_mode(
        decode_mode,
        timestamp_count=len(timestamps),
        frame_count=video_info.frame_count,
    )

    try:
        if decode_mode == "sequential":
            return _read_frames_sequentially(
                capture=capture,
                timestamps=timestamps,
                frame_dir=frame_dir,
                image_ext=image_ext,
                save_images=save_images,
                keep_images=keep_images,
                fps=video_info.fps,
            )

        return _read_frames_by_seeking(
            capture=capture,
            timestamps=timestamps,
            frame_dir=frame_dir,
            image_ext=image_ext,
            save_images=save_images,
            keep_images=keep_images,
        )
    finally:
        capture.release()
