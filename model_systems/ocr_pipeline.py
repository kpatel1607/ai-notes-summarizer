from pathlib import Path
from typing import Any, Dict, List, Tuple
import platform
import traceback

import fitz
from PIL import Image, ImageFilter, ImageOps
import pytesseract

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None


if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


class OCRPipeline:
    def __init__(self):
        self.paddle_ocr = None

        if PaddleOCR is not None:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",
                    show_log=False,
                )
            except Exception:
                print("PaddleOCR initialization failed:")
                traceback.print_exc()
                self.paddle_ocr = None

    def extract_from_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        return {
            "text": text.strip(),
            "source": "plain_text",
            "confidence": 1.0,
            "tables": [],
            "table_count": 0,
        }

    def extract_from_pdf(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        document = fitz.open(file_path)

        extracted_text = ""
        extracted_tables = []

        for page in document:
            page_text = self._extract_page_text_with_layout(page)

            if page_text:
                extracted_text += page_text + "\n\n"

            extracted_tables.extend(self._extract_pdf_tables(page))

        document.close()

        cleaned = self.clean_text(extracted_text)

        return {
            "text": cleaned,
            "source": "pdf_text",
            "confidence": 0.94 if cleaned else 0.4,
            "tables": extracted_tables,
            "table_count": len(extracted_tables),
        }

    def extract_from_scanned_pdf(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        document = fitz.open(file_path)

        all_text = ""
        page_confidences = []

        for page_index, page in enumerate(document):
            pix = page.get_pixmap(dpi=300)
            temp_image_path = path.parent / f"_temp_pdf_page_{page_index}.png"

            pix.save(str(temp_image_path))

            page_result = self.extract_from_image(str(temp_image_path))
            page_confidences.append(page_result.get("confidence", 0.0))

            if page_result.get("text"):
                all_text += (
                    f"\n\n--- Page {page_index + 1} ---\n"
                    + page_result["text"]
                )

            self._safe_unlink(temp_image_path)

        document.close()

        cleaned = self.clean_text(all_text)
        confidence = (
            round(sum(page_confidences) / len(page_confidences), 3)
            if page_confidences
            else 0.0
        )

        return {
            "text": cleaned,
            "source": "scanned_pdf_ocr",
            "confidence": confidence if cleaned else 0.25,
            "tables": [],
            "table_count": 0,
        }

    def extract_from_image(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        processed_path = self._preprocess_image(path)

        if self.paddle_ocr is not None:
            try:
                result = self.paddle_ocr.ocr(str(processed_path))
                extracted_lines = self._lines_from_paddle_result(result)
                confidence = self._confidence_from_paddle_result(result)
                cleaned = self.clean_text("\n".join(extracted_lines))

                if cleaned and confidence >= 0.55:
                    self._safe_unlink_if_temp(processed_path, path)
                    return {
                        "text": cleaned,
                        "source": "paddleocr_preprocessed",
                        "confidence": confidence,
                        "tables": [],
                        "table_count": 0,
                    }

            except Exception:
                print("\nPADDLE OCR ERROR:")
                traceback.print_exc()

        try:
            image = Image.open(processed_path)
            fallback_text, confidence = self._extract_with_tesseract_data(
                image,
            )
            cleaned = self.clean_text(fallback_text)

            self._safe_unlink_if_temp(processed_path, path)

            return {
                "text": cleaned,
                "source": "tesseract_layout_fallback",
                "confidence": confidence if cleaned else 0.3,
                "tables": [],
                "table_count": 0,
            }

        except Exception as e:
            self._safe_unlink_if_temp(processed_path, path)
            print(f"Tesseract failed: {e}")

            return {
                "text": "",
                "source": "ocr_failed",
                "confidence": 0.0,
                "error": str(e),
                "tables": [],
                "table_count": 0,
            }

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
