# default.yaml 配置说明与工业化建议

本文件说明 `configs/default.yaml` 中每个参数的用途，以及在“准确率”和“速度”之间如何取舍。

这里说的“工业水准”不是一个固定参数，而是一组目标：

- 准确率：尽量不漏短暂出现的文字，减少 OCR 噪声和错误合并。
- 速度：GPU 环境尽量接近或快于视频时长，CPU 环境尽量控制在可接受范围。
- 可追踪：每次运行都能通过 report 判断耗时、帧数、OCR 统计和异常。
- 可回退：调试配置和批量生产配置分开，不用靠猜。

## 推荐配置

### 1. 批量生产均衡配置

适合大多数短视频、口播视频、商品视频、整页文案审核。

```yaml
sampling:
  strategy: adaptive_boundary
  coarse_interval_seconds: 0.5
  refine_window_seconds: 0.5
  refine_interval_seconds: 0.2
  refine_ocr_batch_size: 1
  refine_max_extra_ocr_frames: 80
  decode_mode: seek
  save_frame_images: false
  keep_frame_images_in_memory: true

ocr:
  device: gpu:0
  batch_size: 8
  use_doc_orientation_classify: false
  use_doc_unwarping: false
  use_textline_orientation: false

debug:
  enabled: false

merge:
  strategy: line_state_machine
  max_gap_seconds: 1.5
  use_position: true
```

说明：

- `debug.enabled: false` 是批量速度的关键。开启 debug 会保存图片和画框，明显增加 I/O。
- `keep_frame_images_in_memory: true` 可以避免“写图片再读图片”的磁盘损耗。
- `ocr.batch_size: 8` 通常适合 GPU；显存不足时降到 `4` 或 `2`。
- `adaptive_boundary` 会增加边界细化 OCR，但 SRT 时间更准。

### 2. 准确率优先配置

适合文字出现很短、字幕切换很快、审核漏检成本较高的场景。

```yaml
sampling:
  strategy: adaptive_boundary
  coarse_interval_seconds: 0.25
  refine_window_seconds: 0.5
  refine_interval_seconds: 0.1
  refine_max_extra_ocr_frames: 160

filter:
  min_confidence: 0.45

merge:
  max_gap_seconds: 1.0
  line_text_similarity_threshold: 0.8
```

代价：

- OCR 帧数会显著增加，速度会下降。
- `min_confidence` 降低后，噪声也会增加，需要结合 `debug` 抽样检查。

### 3. 速度优先配置

适合快速粗筛、低风险审核、长视频预处理。

```yaml
sampling:
  strategy: fixed
  interval_seconds: 0.75
  save_frame_images: false
  keep_frame_images_in_memory: true

ocr:
  batch_size: 8

debug:
  enabled: false
```

代价：

- 时间边界会变粗。
- 低于采样间隔的短暂文字可能漏掉。

### 4. 调试观察配置

适合排查 OCR 框、过滤规则、文字区域、置信度阈值。

```yaml
sampling:
  image_ext: png

debug:
  enabled: true
  draw_rejected: true
```

说明：

- `png` 适合人工观察，不会引入 JPEG 压缩。
- 不建议批量生产时开启 debug。

## 参数详解

## input

### `input.video_path`

单视频默认输入路径。

- 单视频运行时，如果没有传 `--video`，会使用这个路径。
- 批量模式 `--video-dir` 不使用该参数。

推荐：

```yaml
input:
  video_path: "E:/OCR/test_vedio/1.mp4"
```

## batch

批量处理配置，只在使用 `--video-dir` 时生效。

### `batch.recursive`

是否递归扫描子目录。

- `false`：只扫描目标目录第一层。
- `true`：扫描所有子目录。

推荐：

- 视频都放在同一层：`false`
- 视频按项目/日期分文件夹：`true`

### `batch.copy_srt_to_video_dir`

批量模式是否把生成的 `.srt` 复制回原视频目录。

- `true`：输出目录保留一份，视频目录也复制一份。
- `false`：只保存在输出目录。

推荐：

- 需要人工播放检查：`true`
- 输出目录由后续程序消费：`false`

### `batch.video_extensions`

批量扫描的视频扩展名列表。

默认包含：

```yaml
- .mp4
- .mov
- .mkv
- .avi
- .wmv
- .flv
- .webm
- .m4v
```

## sampling

抽帧配置，直接决定“看多少帧”和“时间边界多准”。

### `sampling.strategy`

抽帧策略。

- `fixed`：只按固定间隔抽帧，速度快，时间边界较粗。
- `adaptive_boundary`：先粗采样，再对合并片段边界做细化，时间更准但更慢。

推荐：

- 正式输出 SRT：`adaptive_boundary`
- 快速粗筛：`fixed`

### `sampling.interval_seconds`

固定抽帧间隔，`strategy: fixed` 时使用。

例子：

- `1.0`：每秒 1 帧。
- `0.5`：每秒 2 帧。
- `0.25`：每秒 4 帧。

影响：

- 越小越不容易漏短文字，但 OCR 次数越多。
- 越大速度越快，但容易漏短暂文字。

### `sampling.coarse_interval_seconds`

自适应边界模式下的粗采样间隔。

推荐：

- 均衡：`0.5`
- 准确率优先：`0.25`
- 速度优先：`0.75` 或 `1.0`

### `sampling.refine_window_seconds`

边界细化搜索窗口。

例如粗采样判断文字从 `2.0s` 开始，窗口为 `0.5` 时，会向前搜索到 `1.5s`。

推荐：

- 一般视频：`0.5`
- 抽帧间隔很大时：可增大到 `1.0`

### `sampling.refine_interval_seconds`

边界细化时的密集采样间隔。

推荐：

- 均衡：`0.2`
- 时间精度优先：`0.1`
- 速度优先：`0.25` 或关闭 `adaptive_boundary`

### `sampling.refine_ocr_batch_size`

边界细化阶段额外 OCR 的批大小。

当前推荐保持：

```yaml
refine_ocr_batch_size: 1
```

原因：

- 边界细化是“边扫描边判断”，过大的批量可能多做无用 OCR。
- 实测在当前项目中，边界细化批量大于 1 不一定更快。

### `sampling.refine_max_extra_ocr_frames`

边界细化最多允许额外 OCR 多少帧。

作用：

- 防止长视频或大量片段导致边界细化成本失控。

推荐：

- 短视频：`80`
- 准确率优先：`120` 到 `200`
- 长视频批量：根据时长调低或关闭 `adaptive_boundary`

### `sampling.boundary_text_similarity_threshold`

边界细化时，判断候选帧是否仍然属于同一段文字的文本相似度阈值。

推荐：

- 一般：`0.85`
- OCR 抖动明显：`0.8`
- 错误合并较多：`0.9`

### `sampling.decode_mode`

OpenCV 抽帧解码方式。

- `seek`：按时间戳跳转读取。适合稀疏采样，当前默认推荐。
- `sequential`：顺序解码，适合非常密集采样。
- `auto`：按采样密度自动选择。

当前推荐：

```yaml
decode_mode: seek
```

### `sampling.save_frame_images`

是否保存抽帧图片。

- `false`：正式 OCR 推荐，不落盘，速度更快。
- `true`：需要人工检查抽帧图片时使用。

注意：

- 如果 `debug.enabled: true`，主流程仍会保存调试图片。

### `sampling.keep_frame_images_in_memory`

是否把抽帧保留在内存中直接传给 OCR。

推荐：

```yaml
keep_frame_images_in_memory: true
```

作用：

- 避免“保存图片到磁盘，再让 OCR 读回图片”。
- 对速度和图片质量都有好处。

### `sampling.max_frames`

最多处理多少帧。

- `null`：处理完整视频。
- 数字：用于快速测试。

推荐：

- 正式处理：`null`
- 调试：`5`、`20`、`50`

### `sampling.image_ext`

保存抽帧或 debug 图片时使用的图片格式。

- `png`：无损，适合检查 OCR 画框和小字。
- `jpg`：体积小，但可能有压缩损失。

推荐：

- 调试准确率：`png`
- 大批量保存图片：`jpg`
- 正式 OCR 且不保存图片：影响不大。

## ocr

OCR 引擎配置。

### `ocr.engine`

OCR 引擎名称。

当前项目使用：

```yaml
engine: paddleocr
```

### `ocr.device`

推理设备。

- `gpu:0`：使用第 0 张 GPU。
- `cpu`：使用 CPU。

推荐：

- 工业批量：`gpu:0`
- 没有 GPU 或排查环境：`cpu`

### `ocr.batch_size`

主 OCR 阶段一次送入 PaddleOCR 的帧数。

推荐：

- GPU 8GB 以上：`8`
- 显存不足：`4` 或 `2`
- CPU：`1` 或 `2`
- 需要启用相似帧缓存：`1`

影响：

- 过小：GPU 利用率不足。
- 过大：显存压力增加，可能变慢或报错。

### `ocr.text_detection_model_name`

文字检测模型，负责找文字框。

当前项目使用 PP-OCRv6 medium：

```yaml
text_detection_model_name: PP-OCRv6_medium_det
```

### `ocr.text_recognition_model_name`

文字识别模型，负责把文字框中的图像识别成文本。

当前项目使用：

```yaml
text_recognition_model_name: PP-OCRv6_medium_rec
```

### `ocr.use_doc_orientation_classify`

是否启用文档方向分类。

推荐：

- 普通短视频：`false`
- 大量旋转页面、扫描文档：可尝试 `true`

代价：

- 会增加推理耗时。

### `ocr.use_doc_unwarping`

是否启用文档弯曲矫正。

推荐：

- 普通视频：`false`
- 拍摄纸质文档、弯曲页面：可尝试 `true`

代价：

- 明显增加耗时。

### `ocr.use_textline_orientation`

是否启用文本行方向判断。

推荐：

- 横向字幕和常规画面文字：`false`
- 大量竖排、旋转文字：可尝试 `true`

### `ocr.text_score_threshold`

项目解析 PaddleOCR 结果后的二次识别分数阈值。

当前推荐保持：

```yaml
text_score_threshold: 0.0
```

原因：

- 先保留 OCR 原始输出，再由 `filter.min_confidence` 做统一过滤，更方便调试。

### `ocr.text_det_limit_side_len`

PaddleOCR 检测阶段输入图像的边长限制。

当前模糊/小字场景推荐：

```yaml
text_det_limit_side_len: 1280
```

调参方向：

- 提高到 `1536`：可能提升小字和模糊字召回，但更慢、更占显存。
- 降低到 `960`：速度更快，但小字更容易漏检。

### `ocr.text_det_limit_type`

配合 `text_det_limit_side_len` 使用，决定边长限制方式。

常用值：

- `max`：限制最长边，适合常规视频，速度更稳。
- `min`：让短边达到指定尺寸，可能放大小图，对小字更敏感，但速度成本更高。

当前推荐：

```yaml
text_det_limit_type: max
```

如果视频分辨率较低且文字很小，可以单独实验：

```yaml
text_det_limit_type: min
text_det_limit_side_len: 960
```

### `ocr.text_det_thresh`

文字检测像素级阈值。

当前模糊场景推荐：

```yaml
text_det_thresh: 0.25
```

调参方向：

- 降低：更容易找到弱文字区域，召回更高，但噪声更多。
- 提高：误检更少，但模糊文字更容易漏。

建议范围：

```text
0.2 - 0.35
```

### `ocr.text_det_box_thresh`

文字框置信度阈值。

当前模糊场景推荐：

```yaml
text_det_box_thresh: 0.45
```

调参方向：

- 降低到 `0.35`：更激进，适合漏检严重时测试。
- 提高到 `0.6`：更保守，适合误检太多时测试。

### `ocr.text_det_unclip_ratio`

检测框外扩比例。

作用：

- 模糊文字边缘不稳定时，适当外扩可以让识别阶段拿到更完整的文字区域。

当前推荐：

```yaml
text_det_unclip_ratio: 2.0
```

调参方向：

- `1.6` 到 `2.0`：常规范围。
- `2.2`：文字框截断时可尝试。
- 过大可能把相邻文字或背景噪声合进来。

### `ocr.text_rec_score_thresh`

PaddleOCR 内部识别分数阈值。

当前推荐：

```yaml
text_rec_score_thresh: 0.0
```

原因：

- 模糊视频中低分识别也可能有审核价值。
- 项目后面还有 `filter.min_confidence`，更适合统一控制最终保留结果。

如果误识别太多，可以提高到：

```yaml
text_rec_score_thresh: 0.3
```

### `ocr.return_word_box`

是否返回词级别位置框。

当前推荐：

```yaml
return_word_box: false
```

原因：

- 当前项目主要使用行级文字框。
- 开启后输出更细，但数据量和处理复杂度会增加。

## ocr_cache

相似帧 OCR 复用。

### `ocr_cache.enabled`

是否启用相似帧复用。

推荐：

- 普通短视频：`false`
- PPT、录屏、静态页面：可设为 `true`

注意：

- 启用时建议 `ocr.batch_size: 1`。
- 批量 OCR 和相似帧缓存目前不同时使用。

### `ocr_cache.image_diff_threshold`

相邻帧缩略图平均差异阈值。

- 越小越保守，不容易误复用。
- 越大复用更多，但可能漏掉小变化。

### `ocr_cache.max_pixel_diff_threshold`

相邻帧缩略图最大像素差异阈值。

作用：

- 防止一个很小的新文字被平均差异稀释。

### `ocr_cache.image_size`

生成帧指纹时缩放到的尺寸。

推荐：

- `96`：当前均衡值。
- 更大：更敏感但稍慢。
- 更小：更快但可能忽略小变化。

## filter

OCR 后处理过滤。

### `filter.min_confidence`

最低 OCR 置信度。

推荐：

- 均衡：`0.5` 到 `0.6`
- 准确率/召回优先：`0.45`
- 降低噪声：`0.65`

当前项目整页审核建议不要设得过高，否则价格、短词、小字可能被误删。

### `filter.noise.enabled`

是否启用明显噪声过滤。

推荐：

```yaml
enabled: true
```

### `filter.noise.min_text_length`

最短文本长度。

推荐：

- 整页审核：`2`
- 如果需要保留单字符编号：`1`

### `filter.noise.drop_symbol_only`

是否丢弃纯符号文本。

推荐：

- 一般：`true`
- 如果符号本身有业务意义：`false`

### `filter.noise.drop_repeated_punctuation`

是否丢弃重复标点。

推荐：

```yaml
drop_repeated_punctuation: true
```

### `filter.noise.drop_numeric_only`

是否丢弃纯数字文本。

整页审核推荐：

```yaml
drop_numeric_only: false
```

原因：

- 价格、折扣、倒计时、编号都可能是重要文字。

### `filter.text_region.enabled`

是否只保留指定区域文字。

当前项目目标是整页文字提取，推荐：

```yaml
enabled: false
```

如果只做字幕区域，可以打开并配置 `x_min/y_min/x_max/y_max`。

### `filter.text_region.x_min/y_min/x_max/y_max`

区域比例坐标，范围 `0.0` 到 `1.0`。

例如只保留下半屏：

```yaml
text_region:
  enabled: true
  x_min: 0.0
  y_min: 0.45
  x_max: 1.0
  y_max: 1.0
```

## layout

版面排序配置。

### `layout.sort_lines`

是否按阅读顺序排序同一帧文字。

推荐：

```yaml
sort_lines: true
```

### `layout.row_tolerance`

判断文字框是否属于同一行的容忍度。

推荐：

- 常规横排文字：`0.6`
- 行距很密：`0.4`
- 字体大小变化大：`0.8`

## debug

调试图片配置。

### `debug.enabled`

是否保存画框调试图片。

推荐：

- 批量生产：`false`
- 规则调试：`true`

注意：

- 开启后会显著增加磁盘 I/O。
- 开启后会保存抽帧图片和 debug 图片。

### `debug.output_dir_name`

debug 图片输出目录名。

默认：

```yaml
output_dir_name: debug
```

### `debug.draw_rejected`

是否把被过滤的 OCR 框也画出来。

推荐调试时：

```yaml
draw_rejected: true
```

这样可以判断是不是过滤规则过严。

### `debug.line_thickness`

画框线宽。

只影响 debug 图片，不影响 OCR。

## merge

跨帧合并配置。

### `merge.strategy`

合并策略。

- `line_state_machine`：轻量状态机，按文本框/文本行跟踪。当前推荐。
- `snapshot`：整帧页面快照合并，输出更像页面级摘要。

推荐：

- 文案审核、需要细粒度：`line_state_machine`
- 想要页面级 SRT：`snapshot`

### `merge.similarity_threshold`

整帧快照合并的文本相似度阈值。

主要用于 `strategy: snapshot`。

推荐：

- 一般：`0.92`
- OCR 抖动明显：`0.85` 到 `0.9`
- 错误合并较多：`0.95`

### `merge.max_gap_seconds`

允许文字短暂漏检的最大间隔。

推荐：

- 一般：`1.5`
- 字幕快速切换：`0.8` 到 `1.0`
- OCR 偶发漏检多：`2.0`

### `merge.use_position`

合并时是否检查文字位置。

推荐：

```yaml
use_position: true
```

原因：

- 防止同样文本在不同位置出现时被误合并。

### `merge.position_match_threshold`

整帧快照策略中，行级位置匹配比例达到多少才合并。

主要用于 `strategy: snapshot`。

推荐：

```yaml
position_match_threshold: 0.5
```

### `merge.line_text_similarity_threshold`

行级文本匹配阈值。

推荐：

- 一般：`0.85`
- OCR 抖动大：`0.8`
- 错误合并多：`0.9`

### `merge.line_iou_threshold`

两个文字框重叠比例阈值。

推荐：

```yaml
line_iou_threshold: 0.3
```

### `merge.line_center_distance_threshold`

两个文字框中心点距离阈值，按画面尺寸归一化。

推荐：

- 常规：`0.08`
- 文字轻微漂移：`0.1`
- 防止误合并：`0.05`

## output

输出配置。

### `output.frames_dir`

抽帧图片根目录。

批量模式下，如果没有传 `--frames-dir`，每个视频的抽帧会放到该视频输出子目录的 `frames/` 下。

### `output.output_dir`

输出根目录。

单视频默认输出到这里。

批量模式下，每个视频会在该目录下建立独立子目录。

### `output.formats`

输出格式列表。

可选：

```yaml
- json
- txt
- srt
```

推荐正式处理保留全部三种。

### `output.report.enabled`

是否输出运行统计报告。

推荐：

```yaml
enabled: true
```

### `output.report.formats`

统计报告格式。

推荐：

```yaml
formats:
  - json
  - txt
```

## 调参顺序建议

如果准确率不够：

1. 先打开 `debug.enabled: true`，检查是否是 OCR 没检测到，还是过滤/合并误删。
2. 如果 OCR 没检测到，减小 `coarse_interval_seconds` 或 `interval_seconds`。
3. 如果时间偏晚，调小 `refine_interval_seconds` 或增大 `refine_max_extra_ocr_frames`。
4. 如果文本被过滤，降低 `filter.min_confidence` 或调整 `filter.noise`。
5. 如果合并错误，调整 `merge.line_text_similarity_threshold`、`merge.line_center_distance_threshold`。

如果速度不够：

1. 关闭 `debug.enabled`。
2. 保持 `save_frame_images: false` 和 `keep_frame_images_in_memory: true`。
3. 增大 `ocr.batch_size`，显存不足再调小。
4. 增大 `coarse_interval_seconds`。
5. 降低或关闭边界细化：`strategy: fixed`。

## 当前项目推荐结论

对当前“整页文字审核”目标，建议默认采用：

```yaml
sampling:
  strategy: adaptive_boundary
  coarse_interval_seconds: 0.5
  refine_interval_seconds: 0.2
  save_frame_images: false
  keep_frame_images_in_memory: true

ocr:
  device: gpu:0
  batch_size: 8
  use_textline_orientation: false

filter:
  min_confidence: 0.5

debug:
  enabled: false

merge:
  strategy: line_state_machine
  use_position: true

output:
  report:
    enabled: true
```

如果要检查图片质量或 OCR 框，再临时开启：

```yaml
debug:
  enabled: true
```
