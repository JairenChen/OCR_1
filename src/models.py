from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SampledFrame:
    index: int
    timestamp_ms: int
    image_path: Path
    image: Any | None = None
    width: int | None = None
    height: int | None = None
    image_saved: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "image_saved": self.image_saved,
        }


@dataclass
class OcrLine:
    text: str
    confidence: float
    polygon: list[list[float]] | None = None
    box: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "polygon": self.polygon,
            "box": self.box,
        }


@dataclass
class FrameOcrResult:
    frame_index: int
    timestamp_ms: int
    image_path: Path
    lines: list[OcrLine]
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "lines": [line.to_dict() for line in self.lines],
        }


@dataclass
class SubtitleSegment:
    start_ms: int
    end_ms: int
    text: str
    confidence: float

    def to_dict(self, start_time: str, end_time: str) -> dict[str, Any]:
        return {
            "start_time": start_time,
            "end_time": end_time,
            "text": self.text,
            "confidence": round(self.confidence, 4),
        }
