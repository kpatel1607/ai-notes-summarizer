from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import platform
import re

import fitz
from PIL import Image, ImageFilter, ImageOps
import pytesseract

from model_systems.extraction_router import ExtractionRouter

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None


if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


class OCRPipeline:
    def __init__(self):
        self.rapid_ocr = None
        self.paddle_ocr = None
        self.enable_paddle = os.getenv(
            "LUMINA_ENABLE_PADDLEOCR",
            "false",
        ).lower().strip() == "true"
        self.extraction_router = ExtractionRouter()

    def _get_rapid_ocr(self):
        if RapidOCR is not None:
            try:
                if self.rapid_ocr is None:
                    self.rapid_ocr = RapidOCR()
            except Exception as exc:
                print("RapidOCR initialization failed:", exc.__class__.__name__)
                self.rapid_ocr = None

        return self.rapid_ocr

    def _get_paddle_ocr(self):
        if not self.enable_paddle:
            return None

        if PaddleOCR is not None:
            try:
                if self.paddle_ocr is None:
                    self.paddle_ocr = PaddleOCR(
                        use_angle_cls=True,
                        lang="en",
                        show_log=False,
                    )
            except Exception as exc:
                print("PaddleOCR initialization failed:", exc.__class__.__name__)
                self.paddle_ocr = None

        return self.paddle_ocr

    def extract_from_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        cleaned = text.strip()
        extracted = self.extraction_router.extract_text(cleaned)

        return {
            **self._normalize_extraction_result(extracted),
            "source": "plain_text",
        }

    def extract_from_pdf(
        self,
        file_path: str,
        route_config: Dict[str, Any] = None,
        features: Dict[str, Any] = None,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        extracted = self.extraction_router.extract_pdf(
            file_path,
            route_config=route_config or {},
            features=features or {},
            mode=mode,
            task=task,
            user_plan=user_plan,
        )
        normalized = self._normalize_extraction_result(extracted)
        normalized["source"] = extracted.get("source", "pdf_text")
        return normalized

    def extract_from_scanned_pdf(
        self,
        file_path: str,
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        extracted = self.extraction_router.scanned_pdf_extractor.extract(
            file_path,
            user_plan=user_plan,
        )
        normalized = self._normalize_extraction_result(extracted)
        normalized["source"] = "scanned_pdf_ocr"
        normalized["page_results"] = extracted.get("page_results", [])
        normalized["pages_processed"] = extracted.get("pages_processed", 0)
        normalized["limit_applied"] = extracted.get("limit_applied", False)
        return normalized

    def extract_from_image(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        extracted = self.extraction_router.image_ocr_extractor.extract(
            file_path,
        )
        normalized = self._normalize_extraction_result(extracted)
        normalized["detected_blocks"] = extracted.get("detected_blocks", [])
        return normalized

    def clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        cleaned = text.replace("\r", "\n")
        cleaned = cleaned.replace("•", "-")
        cleaned = cleaned.replace("·", "-")

        cleaned_lines = []

        for line in cleaned.splitlines():
            normalized = " ".join(line.split()).strip()

            if normalized:
                cleaned_lines.append(normalized)

        return "\n".join(cleaned_lines).strip()

    def _normalize_extraction_result(
        self,
        extracted: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = self.clean_text(
            extracted.get("text")
            or extracted.get("extracted_text")
            or extracted.get("markdown")
            or ""
        )
        confidence = float(
            extracted.get("confidence")
            or extracted.get("ocr_confidence")
            or extracted.get("avg_ocr_confidence")
            or (1.0 if text else 0.0)
        )

        return {
            **extracted,
            "text": text,
            "source": extracted.get("source")
            or extracted.get("extraction_method")
            or "unknown",
            "confidence": round(confidence, 3),
            "quality": self._quality_report(text, confidence),
            "tables": extracted.get("tables", []),
            "table_count": extracted.get(
                "table_count",
                len(extracted.get("tables", [])),
            ),
        }

    def _quality_report(
        self,
        text: str,
        confidence: float,
    ) -> Dict[str, Any]:
        words = text.split()
        word_count = len(words)
        suspicious_words = [
            word
            for word in words
            if self._looks_like_ocr_noise(word)
        ]

        noise_ratio = (
            round(len(suspicious_words) / word_count, 3)
            if word_count
            else 1.0
        )

        if confidence >= 0.85 and noise_ratio <= 0.08 and word_count >= 20:
            label = "high"
        elif confidence >= 0.55 and noise_ratio <= 0.18 and word_count >= 8:
            label = "medium"
        else:
            label = "low"

        return {
            "label": label,
            "word_count": word_count,
            "noise_ratio": noise_ratio,
            "confidence": round(confidence, 3),
            "preview": " ".join(words[:80]),
            "handwriting_experimental": confidence < 0.6 and noise_ratio > 0.12,
        }

    def _looks_like_ocr_noise(
        self,
        word: str,
    ) -> bool:
        cleaned = word.strip()

        if not cleaned:
            return True

        if len(cleaned) <= 2:
            return False

        symbol_count = sum(
            1
            for char in cleaned
            if not char.isalnum()
        )

        if symbol_count / max(len(cleaned), 1) > 0.45:
            return True

        if any(char.isdigit() for char in cleaned) and any(
            char.isalpha()
            for char in cleaned
        ):
            return bool(re.search(r"[A-Za-z]\d|\d[A-Za-z]", cleaned))

        return False

    def _extract_page_text_with_layout(
        self,
        page,
    ) -> str:
        try:
            blocks = page.get_text("blocks")
            text_blocks = []

            for block in sorted(blocks, key=lambda item: (item[1], item[0])):
                block_text = block[4].strip()

                if block_text:
                    text_blocks.append(block_text)

            if text_blocks:
                return "\n".join(text_blocks).strip()
        except Exception:
            pass

        return page.get_text().strip()

    def _extract_pdf_tables(
        self,
        page,
    ) -> List[List[List[str]]]:
        tables = []

        try:
            if not hasattr(page, "find_tables"):
                return tables

            found_tables = page.find_tables()

            for table in found_tables.tables:
                rows = table.extract()
                cleaned_rows = []

                for row in rows:
                    cleaned_row = [
                        self.clean_text(str(cell or ""))
                        for cell in row
                    ]

                    if any(cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if cleaned_rows:
                    tables.append(cleaned_rows)
        except Exception:
            pass

        return tables

    def _preprocess_image(
        self,
        path: Path,
    ) -> Path:
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("L")

            min_width = 1800

            if image.width < min_width:
                scale = min_width / max(image.width, 1)
                image = image.resize(
                    (
                        int(image.width * scale),
                        int(image.height * scale),
                    ),
                    Image.Resampling.LANCZOS,
                )

            image = ImageOps.autocontrast(image)
            image = image.filter(ImageFilter.SHARPEN)

            processed_path = path.with_name(f"{path.stem}_ocr_ready.png")
            image.save(processed_path)

            return processed_path
        except Exception:
            return path

    def _lines_from_paddle_result(
        self,
        result,
    ) -> List[str]:
        positioned_lines = []

        if not result or not result[0]:
            return []

        for line in result[0]:
            bbox = line[0]
            text = line[1][0]

            if not text.strip():
                continue

            x_values = [point[0] for point in bbox]
            y_values = [point[1] for point in bbox]

            positioned_lines.append(
                {
                    "x": min(x_values),
                    "y": min(y_values),
                    "text": text.strip(),
                }
            )

        positioned_lines.sort(
            key=lambda item: (
                round(item["y"] / 14),
                item["x"],
            )
        )

        return [line["text"] for line in positioned_lines]

    def _lines_from_rapidocr_result(
        self,
        result,
    ) -> List[str]:
        positioned_lines = []

        if not result:
            return []

        for line in result:
            try:
                bbox = line[0]
                text = str(line[1]).strip()
            except Exception:
                continue

            if not text:
                continue

            x_values = [point[0] for point in bbox]
            y_values = [point[1] for point in bbox]

            positioned_lines.append(
                {
                    "x": min(x_values),
                    "y": min(y_values),
                    "text": text,
                }
            )

        positioned_lines.sort(
            key=lambda item: (
                round(item["y"] / 14),
                item["x"],
            )
        )

        return [line["text"] for line in positioned_lines]

    def _confidence_from_paddle_result(
        self,
        result,
    ) -> float:
        scores = []

        if result and result[0]:
            for line in result[0]:
                try:
                    scores.append(float(line[1][1]))
                except Exception:
                    pass

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 3)

    def _confidence_from_rapidocr_result(
        self,
        result,
    ) -> float:
        scores = []

        if result:
            for line in result:
                try:
                    scores.append(float(line[2]))
                except Exception:
                    pass

        if not scores:
            return 0.0

        return round(sum(scores) / len(scores), 3)

    def _extract_with_tesseract_data(
        self,
        image: Image.Image,
    ) -> Tuple[str, float]:
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6",
            )

            rows = {}
            scores = []

            for index, text in enumerate(data.get("text", [])):
                cleaned_text = text.strip()

                if not cleaned_text:
                    continue

                try:
                    confidence = float(data["conf"][index])

                    if confidence >= 0:
                        scores.append(confidence / 100)
                except Exception:
                    pass

                key = (
                    data.get("block_num", [0])[index],
                    data.get("par_num", [0])[index],
                    data.get("line_num", [0])[index],
                )

                rows.setdefault(key, []).append(
                    (
                        data.get("left", [0])[index],
                        cleaned_text,
                    )
                )

            lines = []

            for row in rows.values():
                row.sort(key=lambda item: item[0])
                lines.append(" ".join(item[1] for item in row))

            confidence = (
                round(sum(scores) / len(scores), 3)
                if scores
                else 0.55
            )

            return "\n".join(lines), confidence
        except Exception:
            return pytesseract.image_to_string(image), 0.55

    def _safe_unlink(
        self,
        path: Path,
    ) -> None:
        try:
            path.unlink()
        except Exception:
            pass

    def _safe_unlink_if_temp(
        self,
        processed_path: Path,
        original_path: Path,
    ) -> None:
        if processed_path != original_path:
            self._safe_unlink(processed_path)
