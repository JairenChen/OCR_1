from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import site
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.models import FrameOcrResult, SampledFrame, SubtitleSegment
from src.utils import ensure_dir, format_timestamp
from src.video_reader import VideoInfo


PACKAGE_NAMES = [
    "paddleocr",
    "paddlex",
    "paddlepaddle-gpu",
    "paddlepaddle",
    "torch",
    "torchvision",
    "torchaudio",
    "opencv-python",
    "numpy",
    "shapely",
    "PyYAML",
]


def _round_seconds(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return round(float(value), 4)


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _environment_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "user_site": site.getusersitepackages(),
        "packages": _package_versions(),
    }


def _video_summary(video_info: VideoInfo) -> dict[str, Any]:
    path = video_info.path
    file_size_bytes = path.stat().st_size if path.exists() else 0
    return {
        "path": str(path),
        "file_name": path.name,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 3),
        "fps": round(video_info.fps, 4),
        "frame_count": video_info.frame_count,
        "duration_ms": video_info.duration_ms,
        "duration": format_timestamp(video_info.duration_ms),
        "width": video_info.width,
        "height": video_info.height,
    }


def _sampling_summary(
    sampled_frames: list[SampledFrame],
    strategy: str,
    interval_seconds: float,
    max_frames: int | None,
) -> dict[str, Any]:
    first_ms = sampled_frames[0].timestamp_ms if sampled_frames else None
    last_ms = sampled_frames[-1].timestamp_ms if sampled_frames else None
    return {
        "strategy": strategy,
        "interval_seconds": interval_seconds,
        "max_frames": max_frames,
        "sampled_frames": len(sampled_frames),
        "first_timestamp_ms": first_ms,
        "first_timestamp": format_timestamp(first_ms or 0) if first_ms is not None else None,
        "last_timestamp_ms": last_ms,
        "last_timestamp": format_timestamp(last_ms or 0) if last_ms is not None else None,
    }


def _ocr_summary(frame_results: list[FrameOcrResult]) -> dict[str, Any]:
    line_counts = [len(frame_result.lines) for frame_result in frame_results]
    confidences = [
        float(line.confidence)
        for frame_result in frame_results
        for line in frame_result.lines
    ]
    text_chars = sum(
        len(line.text)
        for frame_result in frame_results
        for line in frame_result.lines
    )
    return {
        "processed_frames": len(frame_results),
        "frames_with_text": sum(1 for count in line_counts if count > 0),
        "total_lines": sum(line_counts),
        "total_text_chars": text_chars,
        "line_count_min": min(line_counts) if line_counts else 0,
        "line_count_max": max(line_counts) if line_counts else 0,
        "line_count_avg": round(_average([float(count) for count in line_counts]), 4),
        "confidence_min": round(min(confidences), 4) if confidences else 0.0,
        "confidence_max": round(max(confidences), 4) if confidences else 0.0,
        "confidence_avg": round(_average(confidences), 4),
    }


def _segment_summary(segments: list[SubtitleSegment]) -> dict[str, Any]:
    confidences = [float(segment.confidence) for segment in segments]
    durations = [max(0, segment.end_ms - segment.start_ms) for segment in segments]
    text_chars = sum(len(segment.text) for segment in segments)
    return {
        "segments": len(segments),
        "total_text_chars": text_chars,
        "total_duration_ms": sum(durations),
        "total_duration": format_timestamp(sum(durations)),
        "duration_avg_ms": round(_average([float(value) for value in durations]), 4),
        "confidence_avg": round(_average(confidences), 4),
        "first_start_time": format_timestamp(segments[0].start_ms) if segments else None,
        "last_end_time": format_timestamp(segments[-1].end_ms) if segments else None,
    }


def build_run_report(
    video_info: VideoInfo,
    sampled_frames: list[SampledFrame],
    frame_results: list[FrameOcrResult],
    segments: list[SubtitleSegment],
    written_paths: list[Path],
    timings: dict[str, float],
    settings: dict[str, Any],
) -> dict[str, Any]:
    normalized_timings = {
        key: _round_seconds(value)
        for key, value in timings.items()
    }
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": settings,
        "video": _video_summary(video_info),
        "sampling": _sampling_summary(
            sampled_frames,
            strategy=str(settings.get("sampling", {}).get("strategy", "fixed")),
            interval_seconds=float(settings.get("sampling", {}).get("interval_seconds", 0.0)),
            max_frames=settings.get("sampling", {}).get("max_frames"),
        ),
        "ocr": _ocr_summary(frame_results),
        "ocr_cache": settings.get("ocr_cache", {"enabled": False}),
        "merge": _segment_summary(segments),
        "boundary_refinement": settings.get("boundary_refinement", {"enabled": False}),
        "timings_seconds": normalized_timings,
        "environment": _environment_summary(),
        "outputs": [str(path) for path in written_paths],
    }


def format_run_report_text(report: dict[str, Any]) -> str:
    video = report["video"]
    sampling = report["sampling"]
    ocr = report["ocr"]
    ocr_cache = report.get("ocr_cache", {"enabled": False})
    merge = report["merge"]
    boundary_refinement = report.get("boundary_refinement", {"enabled": False})
    timings = report["timings_seconds"]
    environment = report["environment"]
    settings = report["settings"]
    sampling_settings = settings.get("sampling", {})
    ocr_settings = settings.get("ocr", {})
    packages = environment["packages"]

    lines = [
        "# 运行统计报告",
        "",
        f"生成时间: {report['created_at']}",
        f"命令: {settings.get('command', '')}",
        f"配置文件: {settings.get('config_path', '')}",
        f"运行模式: {'只抽帧' if settings.get('sample_only') else '完整 OCR pipeline'}",
        "",
        "## 视频",
        f"文件: {video['path']}",
        f"大小: {video['file_size_mb']} MB",
        f"分辨率: {video['width']}x{video['height']}",
        f"FPS: {video['fps']}",
        f"总帧数: {video['frame_count']}",
        f"时长: {video['duration']}",
        "",
        "## 抽帧",
        f"策略: {sampling['strategy']}",
        f"间隔: {sampling['interval_seconds']} 秒",
        f"解码模式: {sampling_settings.get('decode_mode')}",
        f"最大帧数限制: {sampling['max_frames']}",
        f"保存抽帧图片: {sampling_settings.get('save_frame_images')}",
        f"保留内存帧: {sampling_settings.get('keep_frame_images_in_memory')}",
        f"实际抽帧数: {sampling['sampled_frames']}",
        f"首帧时间: {sampling['first_timestamp']}",
        f"末帧时间: {sampling['last_timestamp']}",
        "",
        "## OCR 复用缓存",
        f"启用: {ocr_cache.get('enabled')}",
        f"禁用原因: {ocr_cache.get('disabled_reason')}",
        f"缩略图尺寸: {ocr_cache.get('image_size')}",
        f"平均差异阈值: {ocr_cache.get('image_diff_threshold')}",
        f"最大像素差异阈值: {ocr_cache.get('max_pixel_diff_threshold')}",
        f"检查帧数: {ocr_cache.get('checked_frames', 0)}",
        f"实际 OCR 帧数: {ocr_cache.get('ocr_frames', 0)}",
        f"复用帧数: {ocr_cache.get('reused_frames', 0)}",
        f"复用率: {ocr_cache.get('reuse_rate', 0.0)}",
        f"平均差异: min={ocr_cache.get('diff_min')}, avg={ocr_cache.get('diff_avg')}, max={ocr_cache.get('diff_max')}",
        f"最大像素差异: min={ocr_cache.get('max_pixel_diff_min')}, avg={ocr_cache.get('max_pixel_diff_avg')}, max={ocr_cache.get('max_pixel_diff_max')}",
        "",
        "## 边界细化",
        f"启用: {boundary_refinement.get('enabled')}",
        f"细化窗口: {boundary_refinement.get('refine_window_seconds')} 秒",
        f"细化间隔: {boundary_refinement.get('refine_interval_seconds')} 秒",
        f"细化 OCR 批大小: {boundary_refinement.get('ocr_batch_size')}",
        f"参与片段数: {boundary_refinement.get('segment_count', 0)}",
        f"修正片段数: {boundary_refinement.get('refined_segments', 0)}",
        f"开始时间修正数: {boundary_refinement.get('start_updates', 0)}",
        f"结束时间修正数: {boundary_refinement.get('end_updates', 0)}",
        f"额外抽帧数: {boundary_refinement.get('extra_sampled_frames', 0)}",
        f"额外 OCR 帧数: {boundary_refinement.get('extra_ocr_frames', 0)}",
        f"额外 OCR 预算: {boundary_refinement.get('max_extra_ocr_frames')}",
        f"预算跳过次数: {boundary_refinement.get('skipped_by_budget', 0)}",
        "",
        "## OCR 结果",
        f"处理帧数: {ocr['processed_frames']}",
        f"有文字帧数: {ocr['frames_with_text']}",
        f"文字框总数: {ocr['total_lines']}",
        f"文本字符数: {ocr['total_text_chars']}",
        f"每帧文字框: min={ocr['line_count_min']}, avg={ocr['line_count_avg']}, max={ocr['line_count_max']}",
        f"置信度: min={ocr['confidence_min']}, avg={ocr['confidence_avg']}, max={ocr['confidence_max']}",
        "",
        "## OCR 参数",
        f"设备: {ocr_settings.get('device')}",
        f"批大小: {ocr_settings.get('batch_size')}",
        f"检测模型: {ocr_settings.get('text_detection_model_name')}",
        f"识别模型: {ocr_settings.get('text_recognition_model_name')}",
        f"检测边长限制: {ocr_settings.get('text_det_limit_side_len')}",
        f"检测边长类型: {ocr_settings.get('text_det_limit_type')}",
        f"检测像素阈值: {ocr_settings.get('text_det_thresh')}",
        f"检测框阈值: {ocr_settings.get('text_det_box_thresh')}",
        f"检测框外扩比例: {ocr_settings.get('text_det_unclip_ratio')}",
        f"识别内部阈值: {ocr_settings.get('text_rec_score_thresh')}",
        f"项目识别分数阈值: {ocr_settings.get('text_score_threshold')}",
        f"返回词级框: {ocr_settings.get('return_word_box')}",
        "",
        "## 合并结果",
        f"合并策略: {settings.get('merge', {}).get('strategy')}",
        f"片段数: {merge['segments']}",
        f"合并后文本字符数: {merge['total_text_chars']}",
        f"片段总覆盖时长: {merge['total_duration']}",
        f"平均片段时长(ms): {merge['duration_avg_ms']}",
        f"平均置信度: {merge['confidence_avg']}",
        f"首段开始: {merge['first_start_time']}",
        f"末段结束: {merge['last_end_time']}",
        "",
        "## 用时",
        f"视频信息读取: {timings.get('video_info_seconds', 0.0)} 秒",
        f"抽帧: {timings.get('sampling_seconds', 0.0)} 秒",
        f"OCR/过滤/debug: {timings.get('ocr_filter_debug_seconds', 0.0)} 秒",
        f"跨帧合并: {timings.get('merge_seconds', 0.0)} 秒",
        f"边界细化: {timings.get('boundary_refinement_seconds', 0.0)} 秒",
        f"结果导出: {timings.get('export_seconds', 0.0)} 秒",
        f"总用时: {timings.get('total_seconds', 0.0)} 秒",
        "",
        "## 环境",
        f"系统: {environment['platform']}",
        f"Python: {environment['python_version']}",
        f"Python 路径: {environment['python_executable']}",
        f"工作目录: {environment['cwd']}",
        f"Conda 环境: {environment['conda_default_env']}",
        f"Conda 路径: {environment['conda_prefix']}",
        f"PYTHONNOUSERSITE: {environment['python_no_user_site']}",
        f"CUDA_VISIBLE_DEVICES: {environment['cuda_visible_devices']}",
        f"OCR 设备配置: {ocr_settings.get('device')}",
        f"OCR 批大小: {ocr_settings.get('batch_size')}",
        f"PaddleOCR: {packages.get('paddleocr')}",
        f"PaddlePaddle GPU: {packages.get('paddlepaddle-gpu')}",
        f"Torch: {packages.get('torch')}",
        f"OpenCV: {packages.get('opencv-python')}",
        f"NumPy: {packages.get('numpy')}",
        "",
        "## 输出文件",
    ]
    lines.extend(f"- {path}" for path in report["outputs"])
    lines.append("")
    return "\n".join(lines)


def export_run_report_json(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return output_path


def export_run_report_txt(path: str | Path, report: dict[str, Any]) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        file.write(format_run_report_text(report))
    return output_path
