import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz

from model_systems.extraction.image_ocr_extractor import ImageOCRExtractor


class ScannedPDFExtractor:
    def __init__(self, image_ocr: Optional[ImageOCRExtractor] = None):
        self.image_ocr = image_ocr or ImageOCRExtractor()
        self.default_dpi = int(os.getenv("LUMINA_SCANNED_PDF_DPI", "175"))
        self.retry_dpi = int(os.getenv("LUMINA_SCANNED_PDF_RETRY_DPI", "225"))
        self.free_page_limit = int(os.getenv("LUMINA_FREE_OCR_PAGE_LIMIT", "8"))

    def extract(
        self,
        file_path: str,
        *,
        user_plan: str = "free",
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        page_results: List[Dict[str, Any]] = []
        page_texts: List[str] = []
        confidence_scores: List[float] = []

        with fitz.open(file_path) as document:
            page_count = document.page_count
            limit = max_pages or page_count

            if user_plan == "free":
                limit = min(limit, self.free_page_limit)

            pages_to_process = min(page_count, limit)

            for page_index in range(pages_to_process):
                page = document.load_page(page_index)
                page_result = self._extract_page(page, page_index, self.default_dpi)

                if page_result.get("ocr_confidence", 0.0) < 0.45:
                    page_result = self._extract_page(page, page_index, self.retry_dpi)

                page_results.append(page_result)
                confidence_scores.append(page_result.get("ocr_confidence", 0.0))

                text = page_result.get("extracted_text", "").strip()

                if text:
                    page_texts.append(f"--- Page {page_index + 1} ---\n{text}")

        avg_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 3)
            if confidence_scores
            else 0.0
        )
        extracted_text = "\n\n".join(page_texts).strip()
        page_methods = [
            page.get("extraction_method", "")
            for page in page_results
            if page.get("extraction_method")
        ]
        primary_ocr_method = page_methods[0] if page_methods else "paddleocr"

        return {
            "extracted_text": extracted_text,
            "text": extracted_text,
            "page_results": page_results,
            "avg_ocr_confidence": avg_confidence,
            "ocr_confidence": avg_confidence,
            "confidence": avg_confidence,
            "page_count": page_count,
            "pages_processed": len(page_results),
            "limit_applied": len(page_results) < page_count,
            "extraction_method": f"pymupdf_render+{primary_ocr_method}",
            "source": "scanned_pdf_ocr",
            "tables": [],
            "table_count": 0,
        }

    def _extract_page(self, page, page_index: int, dpi: int) -> Dict[str, Any]:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        temp = tempfile.NamedTemporaryFile(
            suffix=f"_page_{page_index + 1}.png",
            delete=False,
        )
        temp.close()

        try:
            pix.save(temp.name)
            result = self.image_ocr.extract(temp.name)
            result["page_index"] = page_index
            result["dpi"] = dpi
            return result
        finally:
            try:
                Path(temp.name).unlink()
            except Exception:
                pass
