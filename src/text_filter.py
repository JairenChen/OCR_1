from __future__ import annotations

import re
from statistics import median

from src.models import FrameOcrResult, OcrLine
from src.utils import normalize_text


def clean_text(text: str) -> str:
    return normalize_text(text)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", clean_text(text))


def is_symbol_only(text: str) -> bool:
    compacted = compact_text(text)
    if not compacted:
        return False
    return not any(char.isalnum() for char in compacted)


def is_repeated_punctuation(text: str) -> bool:
    compacted = compact_text(text)
    if len(compacted) < 2:
        return False
    if any(char.isalnum() for char in compacted):
        return False
    return len(set(compacted)) <= 2


def is_numeric_only(text: str) -> bool:
    compacted = compact_text(text)
    if not compacted:
        return False
    return all(char.isdigit() or char in ".:,/%+-" for char in compacted) and any(
        char.isdigit() for char in compacted
    )


def get_noise_filter_reason(text: str, noise_config: dict[str, object] | None = None) -> str | None:
    if not noise_config or not noise_config.get("enabled", True):
        return None

    compacted = compact_text(text)
    min_text_length = int(noise_config.get("min_text_length", 1))
    if len(compacted) < min_text_length:
        return "noise_too_short"

    if bool(noise_config.get("drop_symbol_only", True)) and is_symbol_only(compacted):
        return "noise_symbol_only"

    if bool(noise_config.get("drop_repeated_punctuation", True)) and is_repeated_punctuation(compacted):
        return "noise_repeated_punctuation"

    if bool(noise_config.get("drop_numeric_only", False)) and is_numeric_only(compacted):
        return "noise_numeric_only"

    return None


def line_box(line: OcrLine) -> tuple[float, float, float, float] | None:
    if not line.box or len(line.box) < 4:
        return None
    x_min, y_min, x_max, y_max = line.box[:4]
    return float(x_min), float(y_min), float(x_max), float(y_max)


def line_center(line: OcrLine) -> tuple[float, float] | None:
    box = line_box(line)
    if box is None:
        return None
    x_min, y_min, x_max, y_max = box
    return (x_min + x_max) / 2, (y_min + y_max) / 2


def line_height(line: OcrLine) -> float | None:
    box = line_box(line)
    if box is None:
        return None
    return max(0.0, box[3] - box[1])


def sort_ocr_lines_by_layout(lines: list[OcrLine], row_tolerance: float = 0.6) -> list[OcrLine]:
    boxed_lines: list[OcrLine] = []
    unboxed_lines: list[OcrLine] = []
    for line in lines:
        if line_box(line) is None or line_center(line) is None:
            unboxed_lines.append(line)
        else:
            boxed_lines.append(line)

    if not boxed_lines:
        return list(lines)

    heights = [height for line in boxed_lines if (height := line_height(line)) and height > 0]
    row_threshold = max(8.0, median(heights) * row_tolerance) if heights else 12.0
    sorted_by_y = sorted(
        boxed_lines,
        key=lambda line: (
            line_center(line)[1] if line_center(line) else 0.0,
            line_box(line)[0] if line_box(line) else 0.0,
        ),
    )

    rows: list[list[OcrLine]] = []
    row_centers: list[float] = []
    for line in sorted_by_y:
        center = line_center(line)
        if center is None:
            unboxed_lines.append(line)
            continue

        center_y = center[1]
        if not rows or abs(center_y - row_centers[-1]) > row_threshold:
            rows.append([line])
            row_centers.append(center_y)
            continue

        rows[-1].append(line)
        row_centers[-1] = sum(line_center(item)[1] for item in rows[-1] if line_center(item)) / len(rows[-1])

    ordered: list[OcrLine] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda line: line_box(line)[0] if line_box(line) else 0.0))

    return ordered + unboxed_lines


def is_line_in_region(
    line: OcrLine,
    frame_width: int,
    frame_height: int,
    region: dict[str, float],
) -> bool:
    center = line_center(line)
    if center is None:
        return True

    x, y = center
    x_min = float(region.get("x_min", 0.0)) * frame_width
    y_min = float(region.get("y_min", 0.0)) * frame_height
    x_max = float(region.get("x_max", 1.0)) * frame_width
    y_max = float(region.get("y_max", 1.0)) * frame_height
    return x_min <= x <= x_max and y_min <= y <= y_max


def get_filter_reason(
    line: OcrLine,
    min_confidence: float = 0.0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    subtitle_region: dict[str, float] | None = None,
    noise_config: dict[str, object] | None = None,
) -> str | None:
    cleaned = clean_text(line.text)
    if not cleaned:
        return "empty_text"
    if line.confidence < min_confidence:
        return "low_confidence"

    noise_reason = get_noise_filter_reason(cleaned, noise_config=noise_config)
    if noise_reason is not None:
        return noise_reason

    region_enabled = bool(subtitle_region and subtitle_region.get("enabled"))
    if region_enabled and frame_width and frame_height:
        if not is_line_in_region(line, frame_width, frame_height, subtitle_region or {}):
            return "outside_region"

    return None


def filter_ocr_lines(
    lines: list[OcrLine],
    min_confidence: float = 0.0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    subtitle_region: dict[str, float] | None = None,
    noise_config: dict[str, object] | None = None,
    sort_by_layout: bool = True,
    row_tolerance: float = 0.6,
) -> list[OcrLine]:
    filtered: list[OcrLine] = []

    for line in lines:
        cleaned = clean_text(line.text)
        reason = get_filter_reason(
            line,
            min_confidence=min_confidence,
            frame_width=frame_width,
            frame_height=frame_height,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
        )
        if reason is not None:
            continue

        filtered.append(
            OcrLine(
                text=cleaned,
                confidence=line.confidence,
                polygon=line.polygon,
                box=line.box,
            )
        )

    if sort_by_layout:
        return sort_ocr_lines_by_layout(filtered, row_tolerance=row_tolerance)
    return filtered


def filter_frame_result(
    frame_result: FrameOcrResult,
    min_confidence: float = 0.0,
    frame_width: int | None = None,
    frame_height: int | None = None,
    subtitle_region: dict[str, float] | None = None,
    noise_config: dict[str, object] | None = None,
    sort_by_layout: bool = True,
    row_tolerance: float = 0.6,
) -> FrameOcrResult:
    return FrameOcrResult(
        frame_index=frame_result.frame_index,
        timestamp_ms=frame_result.timestamp_ms,
        image_path=frame_result.image_path,
        width=frame_result.width,
        height=frame_result.height,
        lines=filter_ocr_lines(
            frame_result.lines,
            min_confidence=min_confidence,
            frame_width=frame_width,
            frame_height=frame_height,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
            sort_by_layout=sort_by_layout,
            row_tolerance=row_tolerance,
        ),
    )


def frame_text(
    lines: list[OcrLine],
    sort_by_layout: bool = True,
    row_tolerance: float = 0.6,
) -> str:
    ordered_lines = sort_ocr_lines_by_layout(lines, row_tolerance=row_tolerance) if sort_by_layout else lines
    return "\n".join(clean_text(line.text) for line in ordered_lines if clean_text(line.text))
