from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import site
import sys
import types
from typing import Any

from src.models import FrameOcrResult, OcrLine, SampledFrame


class OCRUnavailableError(RuntimeError):
    """Raised when PaddleOCR is not installed or cannot be initialized."""


_DLL_DIRECTORY_HANDLES: list[Any] = []


def add_windows_cuda_dll_directories() -> None:
    if os.name != "nt":
        return

    candidate_dirs: list[Path] = []
    for site_path in site.getsitepackages():
        root = Path(site_path)
        candidate_dirs.extend(root.glob("nvidia/*/bin"))

    for env_path_name in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        env_path = os.environ.get(env_path_name)
        if env_path:
            candidate_dirs.append(Path(env_path) / "Library" / "bin")

    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    existing_path_parts = {part.lower() for part in path_parts}

    for directory in candidate_dirs:
        if not directory.exists():
            continue

        directory_text = str(directory)
        if directory_text.lower() not in existing_path_parts:
            path_parts.insert(0, directory_text)
            existing_path_parts.add(directory_text.lower())

        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
            except OSError:
                pass

    os.environ["PATH"] = os.pathsep.join(path_parts)


def prevent_modelscope_torch_import() -> None:
    """Avoid importing PyTorch through ModelScope while initializing PaddleOCR."""
    if "torch" in sys.modules:
        return
    module_name = "modelscope.utils.torch_utils"
    if module_name in sys.modules:
        return

    torch_utils = types.ModuleType(module_name)
    torch_utils.is_dist = lambda: False
    torch_utils.is_master = lambda: True
    sys.modules[module_name] = torch_utils


def _to_plain_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return []


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _polygon_to_box(polygon: list[Any] | None) -> list[float] | None:
    if not polygon:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for point in polygon:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    if not xs or not ys:
        return None

    return [min(xs), min(ys), max(xs), max(ys)]


def payload_to_ocr_lines(payload: dict[str, Any], score_threshold: float = 0.0) -> list[OcrLine]:
    texts = _to_plain_list(payload.get("rec_texts"))
    scores = _to_plain_list(payload.get("rec_scores"))
    polygons = _to_plain_list(payload.get("rec_polys"))
    boxes = _to_plain_list(payload.get("rec_boxes"))

    lines: list[OcrLine] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue

        confidence = float(scores[index]) if index < len(scores) else 0.0
        if confidence < score_threshold:
            continue

        polygon = _to_jsonable(polygons[index]) if index < len(polygons) else None
        box = boxes[index] if index < len(boxes) else _polygon_to_box(polygon)
        box = [float(item) for item in _to_jsonable(box)] if box else None

        lines.append(
            OcrLine(
                text=text,
                confidence=confidence,
                polygon=polygon,
                box=box,
            )
        )

    return lines


class PaddleOCREngine:
    def __init__(
        self,
        text_detection_model_name: str = "PP-OCRv6_medium_det",
        text_recognition_model_name: str = "PP-OCRv6_medium_rec",
        device: str = "cpu",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = True,
        text_score_threshold: float = 0.0,
        text_det_limit_side_len: int | None = None,
        text_det_limit_type: str | None = None,
        text_det_thresh: float | None = None,
        text_det_box_thresh: float | None = None,
        text_det_unclip_ratio: float | None = None,
        text_rec_score_thresh: float | None = None,
        return_word_box: bool | None = None,
    ) -> None:
        self.text_detection_model_name = text_detection_model_name
        self.text_recognition_model_name = text_recognition_model_name
        self.device = device
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.text_score_threshold = text_score_threshold
        self.text_det_limit_side_len = text_det_limit_side_len
        self.text_det_limit_type = text_det_limit_type
        self.text_det_thresh = text_det_thresh
        self.text_det_box_thresh = text_det_box_thresh
        self.text_det_unclip_ratio = text_det_unclip_ratio
        self.text_rec_score_thresh = text_rec_score_thresh
        self.return_word_box = return_word_box
        self._ocr: Any | None = None

    def _load_model(self) -> Any:
        if self._ocr is not None:
            return self._ocr

        self._guard_against_user_site_package("paddleocr")
        add_windows_cuda_dll_directories()

        try:
            import paddle

            if str(self.device).lower().startswith("gpu"):
                paddle.set_device(self.device)
            prevent_modelscope_torch_import()
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRUnavailableError(
                "PaddleOCR is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            self._ocr = PaddleOCR(
                text_detection_model_name=self.text_detection_model_name,
                text_recognition_model_name=self.text_recognition_model_name,
                use_doc_orientation_classify=self.use_doc_orientation_classify,
                use_doc_unwarping=self.use_doc_unwarping,
                use_textline_orientation=self.use_textline_orientation,
                device=self.device,
            )
        except Exception as exc:  # PaddleOCR can raise several framework-specific errors.
            raise OCRUnavailableError(f"Could not initialize PaddleOCR: {exc}") from exc

        return self._ocr

    @staticmethod
    def _guard_against_user_site_package(package_name: str) -> None:
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.origin is None:
            return

        user_site = site.getusersitepackages()
        origin = str(Path(spec.origin).resolve()).lower()
        prefix = str(Path(sys.prefix).resolve()).lower()
        user_site_path = str(Path(user_site).resolve()).lower()

        if origin.startswith(user_site_path) and not origin.startswith(prefix):
            raise OCRUnavailableError(
                f"{package_name} is being imported from the user site-packages: {spec.origin}\n"
                f"This usually means the conda environment is missing {package_name} and is "
                "falling back to packages outside the environment.\n"
                "Fix for env ocr6:\n"
                "  conda run -n ocr6 python -m pip install --no-user paddleocr==3.7.0 shapely\n"
                "  conda env config vars set -n ocr6 PYTHONNOUSERSITE=1\n"
                "  conda deactivate\n"
                "  conda activate ocr6"
            )

    def predict_image(self, image_path: str | Path) -> list[OcrLine]:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        ocr = self._load_model()
        result = ocr.predict(str(path), **self._predict_kwargs())
        return self._result_to_lines(result)

    def predict_image_data(self, image: Any) -> list[OcrLine]:
        if image is None:
            raise ValueError("image must not be None")

        ocr = self._load_model()
        result = ocr.predict(image, **self._predict_kwargs())
        return self._result_to_lines(result)

    def _result_to_lines(self, result: Any) -> list[OcrLine]:
        lines: list[OcrLine] = []
        for item in result:
            payload = self._extract_payload(item)
            lines.extend(payload_to_ocr_lines(payload, self.text_score_threshold))

        return lines

    def predict_frame(self, frame: SampledFrame) -> FrameOcrResult:
        return FrameOcrResult(
            frame_index=frame.index,
            timestamp_ms=frame.timestamp_ms,
            image_path=frame.image_path,
            lines=(
                self.predict_image_data(frame.image)
                if frame.image is not None
                else self.predict_image(frame.image_path)
            ),
            width=frame.width,
            height=frame.height,
        )

    def predict_frames(self, frames: list[SampledFrame]) -> list[FrameOcrResult]:
        if not frames:
            return []

        inputs = [
            frame.image if frame.image is not None else str(frame.image_path)
            for frame in frames
        ]
        ocr = self._load_model()
        results = list(ocr.predict(inputs, **self._predict_kwargs()))
        if len(results) != len(frames):
            raise ValueError(
                f"PaddleOCR returned {len(results)} results for {len(frames)} input frames."
            )

        frame_results: list[FrameOcrResult] = []
        for frame, item in zip(frames, results):
            payload = self._extract_payload(item)
            frame_results.append(
                FrameOcrResult(
                    frame_index=frame.index,
                    timestamp_ms=frame.timestamp_ms,
                    image_path=frame.image_path,
                    lines=payload_to_ocr_lines(payload, self.text_score_threshold),
                    width=frame.width,
                    height=frame.height,
                )
            )
        return frame_results

    def _predict_kwargs(self) -> dict[str, Any]:
        options = {
            "text_det_limit_side_len": self.text_det_limit_side_len,
            "text_det_limit_type": self.text_det_limit_type,
            "text_det_thresh": self.text_det_thresh,
            "text_det_box_thresh": self.text_det_box_thresh,
            "text_det_unclip_ratio": self.text_det_unclip_ratio,
            "text_rec_score_thresh": self.text_rec_score_thresh,
            "return_word_box": self.return_word_box,
        }
        return {
            key: value
            for key, value in options.items()
            if value is not None
        }

    @staticmethod
    def _extract_payload(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return item.get("res", item)

        json_payload = getattr(item, "json", None)
        if isinstance(json_payload, dict):
            return json_payload.get("res", json_payload)

        raise ValueError(f"Unsupported PaddleOCR result item: {type(item)!r}")
