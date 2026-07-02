import os
from dataclasses import dataclass
from typing import Any, Dict

import fitz
from PIL import Image


@dataclass
class InputAnalysis:
    input_type: str
    confidence: float
    metadata: Dict[str, Any]


class DocumentInputAnalyzer:
    """
    Cheap first-pass input analyzer.

    This module intentionally avoids OCR/model calls. It only inspects text,
    file extension, PDF text density, PDF image presence, and image dimensions.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def analyze_text(self, text: str) -> InputAnalysis:
        cleaned = text.strip()
        words = cleaned.split()
        line_count = len([line for line in cleaned.splitlines() if line.strip()])
        table_like_lines = self._table_like_line_count(cleaned)

        input_type = "plain_text"
        confidence = 0.88

        if table_like_lines >= 3 or line_count > 80:
            input_type = "mixed_document"
            confidence = 0.72

        return InputAnalysis(
            input_type=input_type,
            confidence=confidence,
            metadata={
                "word_count": len(words),
                "line_count": line_count,
                "table_like_lines": table_like_lines,
                "has_tables": table_like_lines >= 3,
            },
        )

    def analyze_file(self, file_path: str) -> InputAnalysis:
        extension = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0

        if extension == ".pdf":
            return self._analyze_pdf(file_path, file_size)

        if extension in self.IMAGE_EXTENSIONS:
            return self._analyze_image(file_path, file_size)

        return InputAnalysis(
            input_type="mixed_document",
            confidence=0.35,
            metadata={
                "extension": extension,
                "file_size": file_size,
            },
        )

    def _analyze_pdf(self, file_path: str, file_size: int) -> InputAnalysis:
        page_count = 0
        word_count = 0
        image_count = 0
        table_like_lines = 0

        with fitz.open(file_path) as document:
            page_count = document.page_count

            for page in document:
                page_text = page.get_text("text").strip()
                word_count += len(page_text.split())
                image_count += len(page.get_images(full=True))
                table_like_lines += self._table_like_line_count(page_text)

        words_per_page = word_count / max(page_count, 1)
        has_images = image_count > 0
        has_tables = table_like_lines >= 3

        if word_count >= 80 and has_images:
            input_type = "mixed_document"
            confidence = 0.82
        elif word_count >= 80:
            input_type = "digital_pdf"
            confidence = 0.91
        elif has_images:
            input_type = "scanned_pdf"
            confidence = 0.84
        else:
            input_type = "scanned_pdf"
            confidence = 0.58

        return InputAnalysis(
            input_type=input_type,
            confidence=confidence,
            metadata={
                "page_count": page_count,
                "file_size": file_size,
                "word_count": word_count,
                "words_per_page": round(words_per_page, 2),
                "image_count": image_count,
                "has_images": has_images,
                "has_tables": has_tables,
                "table_like_lines": table_like_lines,
            },
        )

    def _analyze_image(self, file_path: str, file_size: int) -> InputAnalysis:
        try:
            with Image.open(file_path) as image:
                width, height = image.size
        except Exception:
            width = 0
            height = 0

        return InputAnalysis(
            input_type="image",
            confidence=0.86 if width and height else 0.45,
            metadata={
                "file_size": file_size,
                "width": width,
                "height": height,
                "has_images": True,
                "estimated_ocr_required": True,
            },
        )

    def _table_like_line_count(self, text: str) -> int:
        count = 0

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            has_pipes = stripped.count("|") >= 2
            has_many_spaces = "  " in stripped and len(stripped.split()) >= 4
            has_tabs = "\t" in stripped

            if has_pipes or has_many_spaces or has_tabs:
                count += 1

        return count
