import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps
import pytesseract


try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - optional heavy dependency
    PaddleOCR = None

try:
    import easyocr
except Exception:  # pragma: no cover - optional heavy dependency
    easyocr = None


class ImageOCRExtractor:
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    _paddle_ocr = None
    _easyocr_reader = None

    def __init__(self):
        self.enable_paddle = os.getenv("LUMINA_ENABLE_PADDLEOCR", "true").lower() == "true"
        self.enable_easyocr = os.getenv("LUMINA_ENABLE_EASYOCR", "true").lower() == "true"
        self.max_side = int(os.getenv("LUMINA_OCR_MAX_IMAGE_SIDE", "2200"))

    def extract(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return self._failed("unsupported_image_type")

        processed_path, cleanup = self._prepare_image(path)

        try:
            if self.enable_paddle:
                result = self._extract_with_paddle(processed_path)

                if result.get("extracted_text"):
                    return result

            if self.enable_easyocr:
                result = self._extract_with_easyocr(processed_path)

                if result.get("extracted_text"):
                    return result

            return self._extract_with_tesseract(processed_path)
        finally:
            if cleanup:
                self._safe_unlink(processed_path)

    def _extract_with_paddle(self, image_path: Path) -> Dict[str, Any]:
        ocr = self._get_paddle_ocr()

        if not ocr:
            return self._failed("paddleocr_unavailable")

        try:
            result = ocr.ocr(str(image_path))
            blocks, lines, scores = self._blocks_from_paddle(result)
            confidence = self._avg(scores)

            return self._success(
                text="\n".join(lines),
                blocks=blocks,
                confidence=confidence,
                method="paddleocr",
            )
        except Exception as exc:
            return self._failed(str(exc), method="paddleocr")

    def _extract_with_easyocr(self, image_path: Path) -> Dict[str, Any]:
        reader = self._get_easyocr_reader()

        if not reader:
            return self._failed("easyocr_unavailable")

        try:
            result = reader.readtext(str(image_path), detail=1, paragraph=False)
            blocks = []
            lines = []
            scores = []

            for bbox, text, score in result:
                text = str(text or "").strip()

                if not text:
                    continue

                blocks.append({
                    "text": text,
                    "confidence": float(score),
                    "bbox": bbox,
                })
                lines.append(text)
                scores.append(float(score))

            return self._success(
                text="\n".join(lines),
                blocks=blocks,
                confidence=self._avg(scores),
                method="easyocr",
            )
        except Exception as exc:
            return self._failed(str(exc), method="easyocr")

    def _extract_with_tesseract(self, image_path: Path) -> Dict[str, Any]:
        try:
            with Image.open(image_path) as image:
                data = pytesseract.image_to_data(
                    image,
                    output_type=pytesseract.Output.DICT,
                    config="--oem 3 --psm 6",
                )

            rows = {}
            blocks = []
            scores = []

            for index, text in enumerate(data.get("text", [])):
                cleaned = str(text or "").strip()

                if not cleaned:
                    continue

                confidence = self._safe_confidence(data.get("conf", [])[index])

                if confidence is not None:
                    scores.append(confidence)

                key = (
                    data.get("block_num", [0])[index],
                    data.get("par_num", [0])[index],
                    data.get("line_num", [0])[index],
                )
                rows.setdefault(key, []).append(
                    (
                        data.get("left", [0])[index],
                        cleaned,
                    )
                )
                blocks.append({
                    "text": cleaned,
                    "confidence": confidence or 0.0,
                    "bbox": [
                        data.get("left", [0])[index],
                        data.get("top", [0])[index],
                        data.get("width", [0])[index],
                        data.get("height", [0])[index],
                    ],
                })

            lines = []

            for row in rows.values():
                row.sort(key=lambda item: item[0])
                lines.append(" ".join(item[1] for item in row))

            return self._success(
                text="\n".join(lines),
                blocks=blocks,
                confidence=self._avg(scores) if scores else 0.45,
                method="tesseract",
            )
        except Exception as exc:
            return self._failed(str(exc), method="tesseract")

    def _prepare_image(self, path: Path) -> Tuple[Path, bool]:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

            max_side = max(image.size)

            if max_side <= self.max_side:
                return path, False

            scale = self.max_side / max_side
            resized = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

            temp = tempfile.NamedTemporaryFile(
                suffix=".png",
                delete=False,
            )
            temp.close()
            resized.save(temp.name)

            return Path(temp.name), True

    def _get_paddle_ocr(self):
        if PaddleOCR is None:
            return None

        if ImageOCRExtractor._paddle_ocr is None:
            try:
                ImageOCRExtractor._paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                )
            except Exception as exc:
                print("PaddleOCR unavailable:", str(exc))
                ImageOCRExtractor._paddle_ocr = None

        return ImageOCRExtractor._paddle_ocr

    def _get_easyocr_reader(self):
        if easyocr is None:
            return None

        if ImageOCRExtractor._easyocr_reader is None:
            try:
                ImageOCRExtractor._easyocr_reader = easyocr.Reader(
                    ["en"],
                    gpu=False,
                )
            except Exception as exc:
                print("EasyOCR unavailable:", str(exc))
                ImageOCRExtractor._easyocr_reader = None

        return ImageOCRExtractor._easyocr_reader

    def _blocks_from_paddle(self, result) -> Tuple[List[Dict[str, Any]], List[str], List[float]]:
        blocks = []
        lines = []
        scores = []

        if not result:
            return blocks, lines, scores

        page_result = result[0] if isinstance(result, list) else result

        for item in page_result or []:
            try:
                bbox = item[0]
                text = str(item[1][0] or "").strip()
                score = float(item[1][1])
            except Exception:
                continue

            if not text:
                continue

            blocks.append({
                "text": text,
                "confidence": score,
                "bbox": bbox,
            })
            lines.append(text)
            scores.append(score)

        return blocks, lines, scores

    def _success(
        self,
        *,
        text: str,
        blocks: List[Dict[str, Any]],
        confidence: float,
        method: str,
    ) -> Dict[str, Any]:
        cleaned = self._clean_text(text)

        return {
            "extracted_text": cleaned,
            "text": cleaned,
            "detected_blocks": blocks,
            "ocr_confidence": round(confidence, 3),
            "confidence": round(confidence, 3),
            "extraction_method": method,
            "source": method,
            "tables": [],
            "table_count": 0,
            "error": "" if cleaned else "No OCR text detected",
        }

    def _failed(self, error: str, method: str = "unavailable") -> Dict[str, Any]:
        return {
            "extracted_text": "",
            "text": "",
            "detected_blocks": [],
            "ocr_confidence": 0.0,
            "confidence": 0.0,
            "extraction_method": method,
            "source": method,
            "tables": [],
            "table_count": 0,
            "error": error,
        }

    def _safe_confidence(self, value: Any) -> Optional[float]:
        try:
            confidence = float(value)
        except Exception:
            return None

        if confidence < 0:
            return None

        return confidence / 100 if confidence > 1 else confidence

    def _avg(self, scores: List[float]) -> float:
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _clean_text(self, text: str) -> str:
        lines = []

        for line in (text or "").replace("\r", "\n").splitlines():
            normalized = " ".join(line.split()).strip()

            if normalized:
                lines.append(normalized)

        return "\n".join(lines).strip()

    def _safe_unlink(self, path: Path) -> None:
        try:
            path.unlink()
        except Exception:
            pass
