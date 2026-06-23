# Video Text Extractor

这是一个“视频画面文案提取”项目。当前阶段的默认目标是提取整个画面中的文字，而不是只提取底部字幕。

```text
视频输入 -> 抽帧 -> PP-OCRv6 OCR -> 文本过滤 -> 版面排序 -> 跨帧合并 -> 可选边界细化 -> 导出 JSON / TXT / SRT
```

OCR 使用 PaddleOCR 3.7 发布的 PP-OCRv6_medium：

- 检测模型：`PP-OCRv6_medium_det`
- 识别模型：`PP-OCRv6_medium_rec`

## 安装

建议先创建虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

第一次运行 PaddleOCR 时会自动下载模型，耗时取决于网络环境。

如果使用 GPU，推荐按 [docs/environment_setup.md](E:/OCR/docs/environment_setup.md) 创建 conda 环境。当前项目的推荐环境名是 `ocr-ppocrv6`。

## 运行

只抽帧，不执行 OCR：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --sample-only --max-frames 5
```

执行完整 pipeline：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --max-frames 20
```

默认配置使用 `gpu:0`。如果要强制使用 CPU，可以加 `--device cpu`，但 Windows 上 Paddle CPU 推理可能遇到 oneDNN 兼容问题。

如果有 GPU，可以尝试：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --device gpu:0
```

批量处理目标文件夹下的所有视频：

```powershell
python -m src.main --video-dir E:\OCR\test_vedio --output-dir E:\OCR\data\batch_outputs
```

批量模式会为每个视频建立独立子目录，例如：

```text
E:\OCR\data\batch_outputs\1\
E:\OCR\data\batch_outputs\6870fe07bacc42159a9968441061cbbe\
```

每个子目录中保存该视频的 JSON / TXT / SRT / report，抽帧文件放在该子目录的 `frames/` 下。默认还会把生成的 `.srt` 复制一份到原视频所在目录，便于直接和视频文件放在一起检查。

如果要递归扫描子目录：

```powershell
python -m src.main --video-dir E:\OCR\test_vedio --output-dir E:\OCR\data\batch_outputs --recursive
```

## 输出

默认输出到 `data/outputs/`：

- `*_sampled_frames.json`：抽帧结果和时间戳
- `*_frame_ocr.json`：每一帧的 OCR 文本、位置框、置信度
- `*_segments.json`：合并后的时间轴文本
- `*.txt`：纯文本
- `*.srt`：字幕文件
- `*_report.json`：本次运行的结构化统计报告
- `*_report.txt`：本次运行的可读统计报告
- `debug/<video_name>/`：带 OCR 框和过滤结果的调试图片
- `batch_summary.json` / `batch_summary.txt`：批量模式下的整体处理汇总

`*_segments.json` 示例：

```json
[
  {
    "start_time": "00:00:01.000",
    "end_time": "00:00:03.000",
    "text": "example page text",
    "confidence": 0.92
  }
]
```

## 配置
- 在configs下有更详细的说明。
主要参数在 `configs/default.yaml`：

- `batch.recursive`：批量模式是否递归扫描子目录
- `batch.copy_srt_to_video_dir`：批量模式是否把生成的 SRT 复制回原视频目录
- `batch.video_extensions`：批量模式扫描的视频扩展名列表
- `sampling.interval_seconds`：每隔多少秒抽一帧
- `sampling.strategy`：抽帧策略，`fixed` 为固定间隔，`adaptive_boundary` 会在合并片段边界附近细化时间
- `sampling.coarse_interval_seconds`：自适应边界细化时的粗采样间隔
- `sampling.refine_window_seconds`：边界前后用于细化的搜索窗口
- `sampling.refine_interval_seconds`：边界细化时的密集采样间隔
- `sampling.refine_ocr_batch_size`：边界细化阶段额外 OCR 的小批量大小，默认 `1` 表示按需逐帧
- `sampling.refine_max_extra_ocr_frames`：边界细化最多额外 OCR 多少帧，用于控制耗时
- `sampling.boundary_text_similarity_threshold`：细化帧与目标片段的文本相似度阈值
- `sampling.decode_mode`：抽帧解码方式，`auto` 会按采样密度选择，`sequential` 为顺序解码，`seek` 为按时间戳随机定位
- `sampling.save_frame_images`：是否保存抽帧图片；批量处理时建议保持 `false`
- `sampling.keep_frame_images_in_memory`：是否把抽帧图像直接保存在内存中送入 OCR，减少磁盘 I/O
- `ocr.batch_size`：一次送入 PaddleOCR 的帧数，GPU 环境下可提高吞吐；设为 `1` 时使用逐帧 OCR
- `ocr.text_det_limit_side_len`：PaddleOCR 检测阶段输入边长限制，小字/糊字可适当调大
- `ocr.text_det_thresh`：文字检测像素阈值，降低可提高模糊文字召回但会增加噪声
- `ocr.text_det_box_thresh`：文字框阈值，降低可减少漏检但会增加误检
- `ocr.text_det_unclip_ratio`：检测框外扩比例，文字框截断时可适当调大
- `ocr.text_rec_score_thresh`：PaddleOCR 内部识别分数阈值，模糊视频建议先保持较低
- `ocr_cache.enabled`：是否启用相似帧 OCR 复用，画面几乎不变时跳过重复 OCR，默认关闭
- `ocr_cache.image_diff_threshold`：相邻帧缩略图平均差异阈值，越小越保守
- `ocr_cache.max_pixel_diff_threshold`：相邻帧缩略图最大像素差异阈值，用于避免小文字变化被平均值稀释
- `ocr.text_detection_model_name`：文字检测模型
- `ocr.text_recognition_model_name`：文字识别模型
- `filter.min_confidence`：最低 OCR 置信度
- `filter.noise.enabled`：是否过滤明显噪声文本
- `filter.text_region.enabled`：是否只保留画面指定区域文字，整页提取时保持 `false`
- `layout.sort_lines`：是否按版面阅读顺序排序同一帧文字框
- `layout.row_tolerance`：判断文字框是否属于同一行的容忍度
- `debug.enabled`：是否保存画框调试图片
- `merge.strategy`：跨帧合并策略，`line_state_machine` 为轻量状态机，`snapshot` 为旧的整帧快照合并
- `merge.similarity_threshold`：连续帧文本相似到什么程度才合并
- `merge.use_position`：跨帧合并时是否同时检查文字框位置
- `merge.position_match_threshold`：两帧文字框位置匹配到什么比例才合并
- `output.report.enabled`：是否导出运行统计报告
- `output.report.formats`：统计报告格式，默认同时导出 JSON 和 TXT

## 测试

```powershell
python -m unittest discover -s tests
```

测试不会真正加载 PaddleOCR 大模型，主要验证抽帧、PaddleOCR 输出解析、字幕合并、JSON 导出等基础逻辑。


## 参考

- PaddleOCR PP-OCRv6 文档：https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html
- PP-OCRv6 medium recognition model card：https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec
