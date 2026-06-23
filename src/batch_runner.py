from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils import ensure_dir


DEFAULT_VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
)


@dataclass
class BatchVideoResult:
    video_path: Path
    output_dir: Path
    frames_dir: Path
    status: str
    written_paths: list[Path]
    copied_srt_path: Path | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "frames_dir": str(self.frames_dir),
            "status": self.status,
            "written_paths": [str(path) for path in self.written_paths],
            "copied_srt_path": str(self.copied_srt_path) if self.copied_srt_path else None,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
        }


def parse_video_extensions(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_VIDEO_EXTENSIONS
    if isinstance(value, str):
        raw_items = re.split(r"[,\s]+", value)
    else:
        raw_items = [str(item) for item in value]

    extensions: list[str] = []
    for item in raw_items:
        extension = item.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)
    return tuple(dict.fromkeys(extensions)) or DEFAULT_VIDEO_EXTENSIONS


def discover_video_files(
    video_dir: str | Path,
    extensions: tuple[str, ...] = DEFAULT_VIDEO_EXTENSIONS,
    recursive: bool = False,
) -> list[Path]:
    root = Path(video_dir)
    if not root.exists():
        raise FileNotFoundError(f"Video directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Video path is not a directory: {root}")

    normalized_extensions = {extension.lower() for extension in extensions}
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )


def safe_path_part(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return cleaned or "unnamed"


def build_video_output_dir(
    video_path: str | Path,
    video_root: str | Path,
    output_root: str | Path,
) -> Path:
    video = Path(video_path).resolve()
    root = Path(video_root).resolve()
    try:
        relative = video.relative_to(root).with_suffix("")
    except ValueError:
        relative = Path(video.stem)
    safe_parts = [safe_path_part(part) for part in relative.parts]
    return Path(output_root).joinpath(*safe_parts)


def build_video_frames_dir(
    video_path: str | Path,
    video_root: str | Path,
    frames_root: str | Path | None,
    video_output_dir: str | Path,
) -> Path:
    if frames_root is None:
        return Path(video_output_dir) / "frames"
    return build_video_output_dir(
        video_path=video_path,
        video_root=video_root,
        output_root=frames_root,
    )


def find_srt_output_path(written_paths: list[Path], video_stem: str) -> Path | None:
    expected_name = f"{video_stem}.srt"
    for path in written_paths:
        if Path(path).name == expected_name:
            return Path(path)
    for path in written_paths:
        if Path(path).suffix.lower() == ".srt":
            return Path(path)
    return None


def copy_srt_to_video_dir(srt_path: str | Path, video_path: str | Path) -> Path:
    source = Path(srt_path)
    if not source.exists():
        raise FileNotFoundError(f"SRT file not found: {source}")
    destination = Path(video_path).with_suffix(".srt")
    if source.resolve() == destination.resolve():
        return destination
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    return destination


def export_batch_summary_json(
    path: str | Path,
    results: list[BatchVideoResult],
    total_elapsed_seconds: float,
) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    payload = {
        "total_videos": len(results),
        "success_count": sum(1 for result in results if result.status == "success"),
        "failed_count": sum(1 for result in results if result.status == "failed"),
        "total_elapsed_seconds": round(total_elapsed_seconds, 4),
        "results": [result.to_dict() for result in results],
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return output_path


def export_batch_summary_txt(
    path: str | Path,
    results: list[BatchVideoResult],
    total_elapsed_seconds: float,
) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    success_count = sum(1 for result in results if result.status == "success")
    failed_count = sum(1 for result in results if result.status == "failed")
    lines = [
        "# 批量视频 OCR 汇总",
        "",
        f"总视频数: {len(results)}",
        f"成功: {success_count}",
        f"失败: {failed_count}",
        f"总用时: {round(total_elapsed_seconds, 4)} 秒",
        "",
        "## 明细",
    ]
    for index, result in enumerate(results, start=1):
        lines.extend(
            [
                "",
                f"### {index}. {result.video_path.name}",
                f"状态: {result.status}",
                f"视频: {result.video_path}",
                f"输出目录: {result.output_dir}",
                f"抽帧目录: {result.frames_dir}",
                f"复制 SRT: {result.copied_srt_path}",
                f"用时: {round(result.elapsed_seconds, 4)} 秒",
            ]
        )
        if result.error:
            lines.append(f"错误: {result.error}")
    lines.append("")
    with output_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return output_path
