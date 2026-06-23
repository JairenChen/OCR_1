from __future__ import annotations

import json
from pathlib import Path

from src.models import FrameOcrResult, SampledFrame, SubtitleSegment
from src.utils import ensure_dir, format_timestamp


def export_sampled_frames_json(path: str | Path, sampled_frames: list[SampledFrame]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [sampled_frame.to_dict() for sampled_frame in sampled_frames],
            file,
            ensure_ascii=False,
            indent=2,
        )
    return output_path


def export_frame_results_json(path: str | Path, frame_results: list[FrameOcrResult]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [frame_result.to_dict() for frame_result in frame_results],
            file,
            ensure_ascii=False,
            indent=2,
        )
    return output_path


def export_segments_json(path: str | Path, segments: list[SubtitleSegment]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            [
                segment.to_dict(
                    start_time=format_timestamp(segment.start_ms),
                    end_time=format_timestamp(segment.end_ms),
                )
                for segment in segments
            ],
            file,
            ensure_ascii=False,
            indent=2,
        )
    return output_path


def export_txt(path: str | Path, segments: list[SubtitleSegment]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        for segment in segments:
            file.write(segment.text)
            file.write("\n")
    return output_path


def export_srt(path: str | Path, segments: list[SubtitleSegment]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            file.write(f"{index}\n")
            file.write(
                f"{format_timestamp(segment.start_ms, srt=True)} --> "
                f"{format_timestamp(segment.end_ms, srt=True)}\n"
            )
            file.write(f"{segment.text}\n\n")
    return output_path
