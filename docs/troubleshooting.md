# Troubleshooting

## PaddleOCR 没有安装

现象：

```text
PaddleOCR is not installed. Run: pip install -r requirements.txt
```

处理：

```powershell
pip install -r requirements.txt
```

如果使用 conda 环境，请显式安装到环境里，例如你的环境名是 `ocr6`：

```powershell
conda run -n ocr6 python -m pip install --no-user paddleocr==3.7.0 shapely
```

## conda 环境读到了用户目录里的包

现象：

```text
C:\Users\28188\AppData\Roaming\Python\Python310\site-packages\paddleocr
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
```

原因：

当前 conda 环境没有安装 `paddleocr`，Python 又允许读取用户目录 `site-packages`，于是导入了环境外的旧包。`shapely` 这类带二进制扩展的包和当前环境的 `numpy` 不匹配，就会出现 ABI 兼容性错误。

处理：

```powershell
conda env config vars set -n ocr6 PYTHONNOUSERSITE=1
conda run -n ocr6 python -m pip install --no-user --force-reinstall --no-cache-dir paddleocr==3.7.0 shapely
conda deactivate
conda activate ocr6
```

验证：

```powershell
python -c "import sys, site; print(sys.executable); print(site.ENABLE_USER_SITE); print([p for p in sys.path if 'AppData\\Roaming' in p])"
python -c "import paddleocr; print(paddleocr.__file__)"
```

`paddleocr.__file__` 应该位于：

```text
...\envs\ocr6\Lib\site-packages\paddleocr
```

## 第一次运行很慢

第一次运行 PaddleOCR 会下载 PP-OCRv6 模型，这是正常现象。下载完成后，后续运行会复用本地缓存。

## 视频无法读取

可能原因：

- 视频路径写错
- OpenCV 不支持该视频编码
- 文件损坏

可以先运行：

```powershell
python -m src.main --video path\to\video.mp4 --sample-only --max-frames 5
```

如果抽帧都失败，问题在视频读取阶段，不在 OCR 阶段。

## 识别出了水印或角标

这是 OCR 正常行为：它会尽量识别画面中所有文字。如果当前任务不是整页提取，而是只想要局部区域，可以开启文字区域过滤：

```yaml
filter:
  text_region:
    enabled: true
    x_min: 0.0
    y_min: 0.45
    x_max: 1.0
    y_max: 1.0
```

如果字幕不在画面下方，需要根据实际视频调整区域。

## Debug 图中真正字幕被标红

原因通常有两个：

- 文字框中心点没有落入 `text_region`。
- 该文字的置信度低于 `filter.min_confidence`。

处理：

1. 先把 `filter.min_confidence` 从 `0.5` 降到 `0.3` 观察变化。
2. 再调整 `text_region.y_min`，例如从 `0.45` 改成 `0.35`。
3. 如果左右两侧字幕被过滤，调整 `x_min` 和 `x_max`。

## Debug 图中水印被标绿

说明水印也满足当前过滤规则。处理方式：

- 如果水印在角落，收窄 `x_min/x_max`。
- 如果水印在上方，提高 `text_region.y_min`。
- 如果水印置信度低，适当提高 `filter.min_confidence`。

## 输出里有很多符号或孤立字符

原因：

OCR 有时会把图案边缘、装饰线、按钮图标识别成字符。

处理：

```yaml
filter:
  noise:
    enabled: true
    min_text_length: 2
    drop_symbol_only: true
    drop_repeated_punctuation: true
```

不建议一开始打开 `drop_numeric_only`，因为整页文字提取中数字、价格、百分比和倒计时经常有价值。

## CPU 推理出现 oneDNN 报错

现象：

```text
NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ...
onednn_instruction.cc
```

处理：

当前项目默认使用 `gpu:0`。如果你的机器有 NVIDIA GPU，优先使用 GPU 跑 OCR：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --max-frames 20
```

如果必须使用 CPU，可以后续再单独调整 Paddle/oneDNN 版本或关闭 oneDNN 优化。

## Torch GPU 和 Paddle GPU 在同一进程里冲突

现象：

```text
OSError: [WinError 127] Error loading ... cudnn_cnn64_9.dll
```

原因：

Windows 下 PyTorch GPU 和 PaddlePaddle GPU 都会加载 CUDA/cuDNN DLL。谁先加载、从哪个目录加载，会影响后导入的框架。PaddleOCR 3.7 会通过 PaddleX/ModelScope 间接碰到 Torch，所以本项目在 `src/ocr_engine.py` 里做了两个保护：

- 先加载 Paddle，再导入 PaddleOCR。
- 在 OCR 初始化阶段阻止 ModelScope 为日志功能导入 Torch。

建议：

- 跑 OCR 的进程里不要先 `import torch`。
- Torch 训练/推理和 PaddleOCR 视频 OCR 最好用不同进程调用。
- 验证环境时，Torch 和 Paddle 分成两个命令测，不要写在同一个 `python -c` 里。
