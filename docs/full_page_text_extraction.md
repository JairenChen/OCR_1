# Full Page Text Extraction Design

## 当前目标

当前阶段的目标是从视频画面中提取整页文字，而不是只提取底部字幕。

这意味着 OCR 会尽量保留画面中的所有有效文字，例如：

- 视频字幕
- 商品卖点文案
- 页面标题
- 角标信息
- UI 文案
- 屏幕录制中的按钮、菜单、说明文字

区域过滤仍然保留，但它只是可选工具。默认配置下不会限制 OCR 到某个局部区域。

## 当前 Pipeline

```text
video
  -> sample frames
  -> PP-OCRv6 text detection and recognition
  -> confidence filtering
  -> optional text region filtering
  -> layout ordering
  -> debug image drawing
  -> repeated page snapshot merging
  -> export JSON / TXT / SRT
```

## 为什么要做版面排序

OCR 模型输出的文字框顺序不一定等于人眼阅读顺序。整页文字提取时，这个问题比字幕提取更明显。

例如画面中有两行文字：

```text
Top Left       Top Right
Bottom Left    Bottom Right
```

如果直接使用模型输出顺序，可能变成：

```text
Bottom Right
Top Left
Bottom Left
Top Right
```

所以我们增加了 `layout.sort_lines`：

1. 先根据文字框中心点的 y 坐标分行。
2. 同一行内按 x 坐标从左到右排序。
3. 行与行之间按从上到下排序。

这不是完整的文档版面分析，但对普通视频画面、短视频文案、屏幕录制页面已经足够作为 MVP。

## 行分组的核心参数

配置项：

```yaml
layout:
  sort_lines: true
  row_tolerance: 0.6
```

`row_tolerance` 用来判断两个文字框是否属于同一行。它会结合文字框高度计算行间容忍度。

- 值太小：同一行可能被拆成多行。
- 值太大：上下两行可能被合成一行。

建议先保持默认 `0.6`，只有当输出顺序明显不对时再调。

## 过滤策略

当前默认只做置信度过滤：

```yaml
filter:
  min_confidence: 0.5
  noise:
    enabled: true
    min_text_length: 2
    drop_symbol_only: true
    drop_repeated_punctuation: true
    drop_numeric_only: false
  text_region:
    enabled: false
```

含义：

- `min_confidence`: 低于该置信度的 OCR 文本丢弃。
- `noise`: 过滤明显无意义的噪声文本。
- `text_region.enabled`: 是否只保留某个画面区域。

整页文字提取时，`text_region.enabled` 应保持 `false`。

### 噪声过滤为什么要保守

整页视频 OCR 里，短文本不一定是噪声。例如：

- `50%`
- `OFF`
- `MIN`
- `OK`
- `05:05:60`

这些都可能是有效画面文字。因此当前默认只过滤非常明显的噪声：

- 空文本
- 置信度过低
- 单个孤立字符
- 纯符号
- 重复标点

`drop_numeric_only` 默认关闭，因为倒计时、价格、百分比、编号经常是有用信息。

如果以后只想提取局部区域，例如右侧面板或底部字幕，可以开启：

```yaml
filter:
  text_region:
    enabled: true
    x_min: 0.0
    y_min: 0.5
    x_max: 1.0
    y_max: 1.0
```

坐标是比例，不是像素。`y_min: 0.5` 表示从画面中线开始到底部。

## Debug 图片

debug 图片用于检查过滤和排序前的 OCR 框：

- 绿色框：通过过滤，会进入最终文本。
- 红色框：被过滤，例如低置信度或不在指定区域。
- 蓝色框：可选文本区域；只有开启 `text_region.enabled` 时才显示。

输出目录：

```text
data/outputs/debug/<video_name>/
```

建议每次调参时先跑少量帧：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --max-frames 5
```

然后打开 debug 图片看框是否合理。

## 跨帧合并

视频中同一页文字可能连续出现多帧。我们不会把每一帧都直接输出成最终文本，而是用相似度合并连续重复页面快照。

当前合并逻辑：

1. 每帧先得到排序后的整页文本。
2. 与上一段文本计算相似度。
3. 检查相似文字是否出现在相近位置。
4. 相似度高、位置相近并且时间间隔不大，就合并为同一时间段。
5. 如果文本变化明显，或同样文本移动到了明显不同的位置，就开始一个新片段。

这一步适合处理静态页面、短视频固定文案、字幕停留多帧等场景。

### 位置感知合并

只看文本相似度会有一个问题：同样的词可能出现在不同位置。

例如：

```text
1s: LIMIT 在画面顶部
2s: LIMIT 在画面底部
```

如果只看文本，二者会被合并。但从画面理解角度看，它们可能是两个不同的文字对象。

因此当前合并会同时判断：

```text
文本相似
+ 时间连续
+ 文本框位置相近
```

位置相近有两个判断方式：

- 文本框 IoU 足够高。
- 文本框中心点距离足够近。

配置项：

```yaml
merge:
  use_position: true
  position_match_threshold: 0.5
  line_text_similarity_threshold: 0.85
  line_iou_threshold: 0.3
  line_center_distance_threshold: 0.08
```

参数含义：

- `use_position`: 是否启用位置感知合并。
- `position_match_threshold`: 两帧中有多少比例的文字行匹配，才认为页面位置稳定。
- `line_text_similarity_threshold`: 两个文字框里的文本要多像，才尝试比较位置。
- `line_iou_threshold`: 两个框重叠比例达到多少，认为位置相同。
- `line_center_distance_threshold`: 两个框中心点距离占画面尺寸的比例，越小越严格。

如果画面中文字会明显移动，可以降低 `position_match_threshold` 或关闭 `use_position`。如果同样文字经常在不同区域误合并，可以提高 `position_match_threshold` 或降低 `line_center_distance_threshold`。

## 自适应边界细化

固定间隔抽帧会带来时间误差。例如 `sampling.interval_seconds: 1.0` 时，文字真实在 `1.2s` 出现，但首次抽到它可能是 `2.0s`。这会让 SRT 的开始时间偏晚。

为减少这种误差，项目支持 `adaptive_boundary` 策略：

```yaml
sampling:
  strategy: adaptive_boundary
  coarse_interval_seconds: 0.5
  refine_window_seconds: 0.5
  refine_interval_seconds: 0.2
  refine_max_extra_ocr_frames: 80
  boundary_text_similarity_threshold: 0.85
  decode_mode: seek
  save_frame_images: false
  keep_frame_images_in_memory: true
```

流程：

1. 先按 `coarse_interval_seconds` 粗采样并 OCR。
2. 先用跨帧合并得到粗略片段。
3. 对每个片段的开始和结束附近重新密集抽帧。
4. 对边界帧重新 OCR，并判断是否仍然匹配该片段文本。
5. 用第一个匹配帧修正开始时间，用最后一个匹配帧修正结束时间。

这会增加额外 OCR 次数，所以运行会变慢。运行结束后可以查看 `*_report.txt` 中的“边界细化”部分，重点看：

```text
修正片段数
开始时间修正数
结束时间修正数
额外抽帧数
额外 OCR 帧数
额外 OCR 预算
边界细化耗时
```

如果想对比优化前后的差异，可以把策略改回：

```yaml
sampling:
  strategy: fixed
```

正式批量处理时建议保持 `save_frame_images: false`。这样 OpenCV 解码出的图像会直接进入 PaddleOCR，减少磁盘写入和读取。需要检查 OCR 框时，再把 `debug.enabled` 打开，此时项目会自动保存调试所需的抽帧图片。

正式批量处理时建议保持当前默认的 `decode_mode: seek`。当前每 0.5 秒抽一帧属于稀疏采样，实测随机定位更稳。若以后提高到更密集采样，可以手动设置为 `sequential` 或 `auto` 做性能对比。

## 当前限制

当前版本仍然是规则型 MVP：

- 不做复杂版面分析，例如多栏文档阅读顺序。
- 不判断文字语义是否重要。
- 不训练 YOLO 或其他检测模型。
- 不区分“字幕文字”和“页面 UI 文字”。

如果后续需要精细区分不同文字类型，可以增加分类规则或进入模型检测阶段。

## 推荐调参顺序

1. 先保持整页模式：`text_region.enabled: false`。
2. 查看 `*_frame_ocr.json`，确认 OCR 文本是否完整。
3. 查看 debug 图片，确认文字框是否正确。
4. 如果噪声太多，先调 `filter.min_confidence`。
5. 如果只需要局部内容，再开启 `filter.text_region.enabled`。
6. 如果输出顺序不自然，再调 `layout.row_tolerance`。
7. 如果仍有纯符号或孤立字符，再调 `filter.noise`。
