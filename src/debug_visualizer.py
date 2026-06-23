from __future__ import annotations

from pathlib import Path

import cv2

from src.models import FrameOcrResult, OcrLine
from src.text_filter import get_filter_reason
from src.utils import ensure_dir


KEEP_COLOR = (40, 180, 80)
DROP_COLOR = (40, 40, 220)
REGION_COLOR = (220, 120, 20)
TEXT_COLOR = (255, 255, 255)


def _box_to_ints(line: OcrLine) -> tuple[int, int, int, int] | None:
    if not line.box or len(line.box) < 4:
        return None
    x_min, y_min, x_max, y_max = line.box[:4]
    return int(x_min), int(y_min), int(x_max), int(y_max)


def _region_to_box(
    region: dict[str, float] | None,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int] | None:
    if not region or not region.get("enabled"):
        return None

    return (
        int(float(region.get("x_min", 0.0)) * frame_width),
        int(float(region.get("y_min", 0.0)) * frame_height),
        int(float(region.get("x_max", 1.0)) * frame_width),
        int(float(region.get("y_max", 1.0)) * frame_height),
    )


def draw_debug_image(
    frame_result: FrameOcrResult,
    output_dir: str | Path,
    frame_width: int,
    frame_height: int,
    min_confidence: float,
    subtitle_region: dict[str, float] | None = None,
    noise_config: dict[str, object] | None = None,
    draw_rejected: bool = True,
    line_thickness: int = 2,
) -> Path:
    image = cv2.imread(str(frame_result.image_path))
    if image is None:
        raise ValueError(f"Could not read image for debug drawing: {frame_result.image_path}")

    region_box = _region_to_box(subtitle_region, frame_width, frame_height)
    if region_box is not None:
        cv2.rectangle(image, region_box[:2], region_box[2:], REGION_COLOR, line_thickness)

    for index, line in enumerate(frame_result.lines):
        reason = get_filter_reason(
            line,
            min_confidence=min_confidence,
            frame_width=frame_width,
            frame_height=frame_height,
            subtitle_region=subtitle_region,
            noise_config=noise_config,
        )
        if reason is not None and not draw_rejected:
            continue

        box = _box_to_ints(line)
        if box is None:
            continue

        color = KEEP_COLOR if reason is None else DROP_COLOR
        label = f"keep {line.confidence:.2f}" if reason is None else f"drop {reason}"
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, line_thickness)
        cv2.putText(
            image,
            f"{index}:{label}",
            (x_min, max(16, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    debug_dir = ensure_dir(output_dir)
    output_path = debug_dir / f"debug_{frame_result.frame_index:06d}_{frame_result.timestamp_ms:09d}ms.jpg"
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Could not save debug image: {output_path}")
    return output_path
