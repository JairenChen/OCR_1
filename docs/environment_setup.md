# GPU Environment Setup

本项目建议使用一个全新的 conda 环境。你的机器当前检测到：

- GPU：NVIDIA GeForce RTX 3050 Laptop GPU
- Driver：560.81
- `nvidia-smi` CUDA：12.6

因此推荐安装 CUDA 12.6 对应的 PyTorch 和 PaddlePaddle GPU wheel。

## 一键重建环境

在 PowerShell 中运行：

```powershell
conda deactivate
conda env remove -n ocr-ppocrv6 -y
conda create -n ocr-ppocrv6 python=3.10 pip -y
conda env config vars set -n ocr-ppocrv6 PYTHONNOUSERSITE=1

# 后续安装全部显式写入 ocr-ppocrv6，避免装到系统 Python 或其它环境。
conda run -n ocr-ppocrv6 python -m pip install --upgrade pip setuptools wheel

# PyTorch GPU, CUDA 12.6
conda run -n ocr-ppocrv6 python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# PaddlePaddle GPU, CUDA 12.6
conda run -n ocr-ppocrv6 python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Project dependencies
conda run -n ocr-ppocrv6 python -m pip install -r requirements-gpu-cu126.txt
```

如果你希望 conda 同时安装 `ffmpeg`：

```powershell
conda install -n ocr-ppocrv6 -c conda-forge ffmpeg -y
```

## 确认 pip 装到了 conda 环境

```powershell
conda run -n ocr-ppocrv6 where python
conda run -n ocr-ppocrv6 python -m pip --version
conda run -n ocr-ppocrv6 python -c "import sys; print(sys.executable)"
```

输出路径应该包含类似：

```text
...\anaconda3\envs\ocr-ppocrv6\python.exe
...\anaconda3\envs\ocr-ppocrv6\Lib\site-packages\pip
```

## 验证 GPU

```powershell
conda run -n ocr-ppocrv6 python -c "import torch; print('torch', torch.__version__); print('torch cuda available:', torch.cuda.is_available()); print('torch cuda:', torch.version.cuda); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
conda run -n ocr-ppocrv6 python -c "import paddle; print('paddle', paddle.__version__); paddle.utils.run_check(); print('paddle cuda compiled:', paddle.is_compiled_with_cuda())"
conda run -n ocr-ppocrv6 python -c "from paddleocr import PaddleOCR; print('paddleocr import ok')"
```

不要在同一个验证命令里连续 `import torch; import paddle`。Windows 下两个 GPU 框架可能抢同一组 CUDA/cuDNN DLL，导致后导入的框架报 `WinError 127`。本项目运行 OCR 时会先加载 Paddle，并避免 PaddleOCR 的 ModelScope 依赖间接导入 Torch。

## 运行项目

先只抽帧：

```powershell
conda activate ocr-ppocrv6
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --sample-only --max-frames 5
```

再跑完整 OCR：

```powershell
python -m src.main --video test_vedio\fefdbfb15f9f4b7b9cff165951d5a0fd.mp4 --device gpu:0 --max-frames 10
```

## 为什么不用一个 requirements 一次装完

PyTorch 和 PaddlePaddle 的 GPU wheel 来自不同官方索引：

- PyTorch CUDA 12.6：`https://download.pytorch.org/whl/cu126`
- PaddlePaddle CUDA 12.6：`https://www.paddlepaddle.org.cn/packages/stable/cu126/`

把它们混在同一个普通 `requirements.txt` 里，容易误装 CPU 版或走错索引。所以本项目采用“先安装两个 GPU 框架，再安装项目依赖”的方式。
