from pathlib import Path
from typing import Dict, Any
import platform
import traceback

import fitz
from PIL import Image
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

    # =========================================
    # TEXT INPUT
    # =========================================

    def extract_from_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        return {
            "text": text.strip(),
            "source": "plain_text",
            "confidence": 1.0,
        }

    # =========================================
    # PDF EXTRACTION
    # =========================================

    def extract_from_pdf(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        document = fitz.open(file_path)

        extracted_text = ""

        for page in document:
            page_text = page.get_text().strip()

            if page_text:
                extracted_text += page_text + "\n\n"

        document.close()

        cleaned = self.clean_text(extracted_text)

        return {
            "text": cleaned,
            "source": "pdf_text",
            "confidence": 0.92 if cleaned else 0.4,
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

        for page_index, page in enumerate(document):
            pix = page.get_pixmap(dpi=220)

            temp_image_path = (
                path.parent / f"_temp_pdf_page_{page_index}.png"
            )

            pix.save(str(temp_image_path))

            page_result = self.extract_from_image(
                str(temp_image_path),
            )

            if page_result.get("text"):
                all_text += (
                    f"\n\n--- Page {page_index + 1} ---\n"
                    + page_result["text"]
                )

            try:
                temp_image_path.unlink()
            except Exception:
                pass

        document.close()

        cleaned = self.clean_text(all_text)

        return {
            "text": cleaned,
            "source": "scanned_pdf_ocr",
            "confidence": 0.82 if cleaned else 0.25,
        }

    # =========================================
    # IMAGE OCR
    # =========================================

    def extract_from_image(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        # -------------------------------
        # PaddleOCR primary extraction
        # -------------------------------

        if self.paddle_ocr is not None:
            try:
                result = self.paddle_ocr.ocr(
                    str(path),
                )

                extracted_lines = []

                if result and result[0]:
                    for line in result[0]:
                        text = line[1][0]

                        if text.strip():
                            extracted_lines.append(text)

                extracted_text = "\n".join(extracted_lines)
                cleaned = self.clean_text(extracted_text)

                if cleaned:
                    return {
                        "text": cleaned,
                        "source": "paddleocr",
                        "confidence": 0.90,
                    }

            except Exception:
                print("\nPADDLE OCR ERROR:")
                traceback.print_exc()

        # -------------------------------
        # Tesseract fallback
        # -------------------------------

        try:
            image = Image.open(path)

            fallback_text = pytesseract.image_to_string(
                image,
            )

            cleaned = self.clean_text(fallback_text)

            return {
                "text": cleaned,
                "source": "tesseract_fallback",
                "confidence": 0.65 if cleaned else 0.3,
            }

        except Exception as e:
            print(f"Tesseract failed: {e}")

            return {
                "text": "",
                "source": "ocr_failed",
                "confidence": 0.0,
                "error": str(e),
            }

    # =========================================
    # CLEAN TEXT
    # =========================================

    def clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        cleaned = text.replace("\r", "\n")

        cleaned = " ".join(cleaned.split())

        cleaned = cleaned.replace("•", "")

        return cleaned.strip()