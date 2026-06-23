from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    duration_ms: int
    width: int
    height: int


def read_video_info(video_path: str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration_ms = int(frame_count / fps * 1000) if fps > 0 and frame_count > 0 else 0
    return VideoInfo(
        path=path,
        fps=fps,
        frame_count=frame_count,
        duration_ms=duration_ms,
        width=width,
        height=height,
    )
