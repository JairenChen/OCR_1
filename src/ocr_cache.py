from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.models import FrameOcrResult, OcrLine, SampledFrame


@dataclass
class OcrReuseStats:
    enabled: bool
    image_diff_threshold: float
    max_pixel_diff_threshold: float
    image_size: int
    checked_frames: int = 0
    reused_frames: int = 0
    ocr_frames: int = 0
    skipped_no_image: int = 0
    skipped_no_reference: int = 0
    disabled_reason: str | None = None
    _diff_total: float = 0.0
    _diff_count: int = 0
    _diff_min: float | None = None
    _diff_max: float | None = None
    _max_pixel_diff_total: float = 0.0
    _max_pixel_diff_min: float | None = None
    _max_pixel_diff_max: float | None = None

    def record_difference(self, metrics: "FrameDifferenceMetrics") -> None:
        self.checked_frames += 1
        self._diff_total += metrics.mean
        self._diff_count += 1
        self._diff_min = metrics.mean if self._diff_min is None else min(self._diff_min, metrics.mean)
        self._diff_max = metrics.mean if self._diff_max is None else max(self._diff_max, metrics.mean)
        self._max_pixel_diff_total += metrics.maximum
        self._max_pixel_diff_min = (
            metrics.maximum
            if self._max_pixel_diff_min is None
            else min(self._max_pixel_diff_min, metrics.maximum)
        )
        self._max_pixel_diff_max = (
            metrics.maximum
            if self._max_pixel_diff_max is None
            else max(self._max_pixel_diff_max, metrics.maximum)
        )

    def to_dict(self) -> dict[str, Any]:
        reuse_rate = self.reused_frames / self.checked_frames if self.checked_frames else 0.0
        diff_avg = self._diff_total / self._diff_count if self._diff_count else 0.0
        max_pixel_diff_avg = (
            self._max_pixel_diff_total / self._diff_count
            if self._diff_count
            else 0.0
        )
        return {
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
            "image_diff_threshold": self.image_diff_threshold,
            "max_pixel_diff_threshold": self.max_pixel_diff_threshold,
            "image_size": self.image_size,
            "checked_frames": self.checked_frames,
            "reused_frames": self.reused_frames,
            "ocr_frames": self.ocr_frames,
            "reuse_rate": round(reuse_rate, 4),
            "skipped_no_image": self.skipped_no_image,
            "skipped_no_reference": self.skipped_no_reference,
            "diff_min": round(self._diff_min, 4) if self._diff_min is not None else None,
            "diff_avg": round(diff_avg, 4),
            "diff_max": round(self._diff_max, 4) if self._diff_max is not None else None,
            "max_pixel_diff_min": (
                round(self._max_pixel_diff_min, 4)
                if self._max_pixel_diff_min is not None
                else None
            ),
            "max_pixel_diff_avg": round(max_pixel_diff_avg, 4),
            "max_pixel_diff_max": (
                round(self._max_pixel_diff_max, 4)
                if self._max_pixel_diff_max is not None
                else None
            ),
        }


@dataclass(frozen=True)
class FrameDifferenceMetrics:
    mean: float
    maximum: float

    def is_reusable(self, mean_threshold: float, max_threshold: float) -> bool:
        return self.mean <= mean_threshold and self.maximum <= max_threshold


def make_frame_fingerprint(image: Any, image_size: int = 64) -> np.ndarray:
    """Convert a full frame into a tiny grayscale image for cheap similarity checks."""
    if image is None:
        raise ValueError("image must not be None")
    if image_size <= 0:
        raise ValueError("image_size must be greater than 0")
    if not hasattr(image, "shape"):
        raise TypeError("image must be a numpy-like image array")

    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) >= 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)


def frame_difference_score(left_fingerprint: np.ndarray, right_fingerprint: np.ndarray) -> float:
    return frame_difference_metrics(left_fingerprint, right_fingerprint).mean


def frame_difference_metrics(
    left_fingerprint: np.ndarray,
    right_fingerprint: np.ndarray,
) -> FrameDifferenceMetrics:
    if left_fingerprint.shape != right_fingerprint.shape:
        raise ValueError(
            f"Fingerprint shapes must match: {left_fingerprint.shape} != {right_fingerprint.shape}"
        )
    diff = cv2.absdiff(left_fingerprint, right_fingerprint)
    return FrameDifferenceMetrics(
        mean=float(np.mean(diff)),
        maximum=float(np.max(diff)),
    )


def clone_ocr_result_for_frame(
    source_result: FrameOcrResult,
    sampled_frame: SampledFrame,
) -> FrameOcrResult:
    """Reuse OCR lines while giving the result the current frame timestamp and path."""
    return FrameOcrResult(
        frame_index=sampled_frame.index,
        timestamp_ms=sampled_frame.timestamp_ms,
        image_path=Path(sampled_frame.image_path),
        width=sampled_frame.width or source_result.width,
        height=sampled_frame.height or source_result.height,
        lines=[
            OcrLine(
                text=line.text,
                confidence=line.confidence,
                polygon=[list(point) for point in line.polygon] if line.polygon else None,
                box=list(line.box) if line.box else None,
            )
            for line in source_result.lines
        ],
    )
