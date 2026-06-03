import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


@dataclass
class InputUnderstandingResult:
    input_type: str
    document_type: str
    extraction_strategy: str
    confidence: float
    metadata: Dict[str, Any]


class InputUnderstandingSystem:
    """
    First-stage Lumina input understanding system.

    Purpose:
    - Detect whether input is plain text, image, digital PDF, scanned PDF, or mixed PDF.
    - Decide the best extraction strategy.
    - Prepare routing for future pretrained model pipelines.
    """

    def analyze_text(
        self,
        text: str,
    ) -> InputUnderstandingResult:
        cleaned = text.strip()

        word_count = len(cleaned.split())
        line_count = len(cleaned.splitlines())

        if word_count < 5:
            document_type = "very_short_text"
            confidence = 0.75
        elif self._looks_like_questions(cleaned):
            document_type = "question_answer_or_exam_text"
            confidence = 0.82
        elif self._looks_like_meeting_notes(cleaned):
            document_type = "professional_notes"
            confidence = 0.78
        else:
            document_type = "general_text"
            confidence = 0.80

        return InputUnderstandingResult(
            input_type="plain_text",
            document_type=document_type,
            extraction_strategy="direct_text_processing",
            confidence=confidence,
            metadata={
                "word_count": word_count,
                "line_count": line_count,
                "has_questions": self._looks_like_questions(cleaned),
            },
        )

    def analyze_file(
        self,
        file_path: str,
    ) -> InputUnderstandingResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            return self._analyze_pdf(file_path)

        if extension in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            return self._analyze_image(file_path)

        return InputUnderstandingResult(
            input_type="unknown_file",
            document_type="unsupported",
            extraction_strategy="reject_or_manual_upload",
            confidence=0.3,
            metadata={
                "extension": extension,
            },
        )

    def _analyze_pdf(
        self,
        file_path: str,
    ) -> InputUnderstandingResult:
        document = fitz.open(file_path)

        page_count = document.page_count
        extracted_text = ""
        image_count = 0

        for page in document:
            extracted_text += page.get_text().strip() + "\n"

            images = page.get_images(full=True)
            image_count += len(images)

        document.close()

        word_count = len(extracted_text.split())

        if word_count > 80:
            input_type = "digital_pdf"
            extraction_strategy = "pdf_text_extraction"
            confidence = 0.9
        elif image_count > 0:
            input_type = "scanned_pdf"
            extraction_strategy = "pdf_to_image_ocr"
            confidence = 0.82
        else:
            input_type = "empty_or_low_text_pdf"
            extraction_strategy = "ocr_fallback"
            confidence = 0.55

        return InputUnderstandingResult(
            input_type=input_type,
            document_type=self._guess_document_type(extracted_text),
            extraction_strategy=extraction_strategy,
            confidence=confidence,
            metadata={
                "page_count": page_count,
                "word_count": word_count,
                "image_count": image_count,
            },
        )

    def _analyze_image(
        self,
        file_path: str,
    ) -> InputUnderstandingResult:
        image = Image.open(file_path)

        width, height = image.size
        aspect_ratio = round(width / height, 2)

        try:
            ocr_text = pytesseract.image_to_string(image)
        except Exception:
            ocr_text = ""

        word_count = len(ocr_text.split())

        if word_count > 30:
            input_type = "text_image"
            extraction_strategy = "image_ocr"
            confidence = 0.82
        elif word_count > 5:
            input_type = "low_text_image"
            extraction_strategy = "enhanced_image_ocr"
            confidence = 0.65
        else:
            input_type = "unclear_image"
            extraction_strategy = "image_preprocessing_then_ocr"
            confidence = 0.45

        return InputUnderstandingResult(
            input_type=input_type,
            document_type=self._guess_document_type(ocr_text),
            extraction_strategy=extraction_strategy,
            confidence=confidence,
            metadata={
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "ocr_word_count": word_count,
                "sample_text": ocr_text[:300],
            },
        )

    def _guess_document_type(
        self,
        text: str,
    ) -> str:
        cleaned = text.lower()

        if not cleaned.strip():
            return "unknown"

        if self._looks_like_questions(cleaned):
            return "question_answer_or_exam_material"

        if self._looks_like_meeting_notes(cleaned):
            return "professional_or_meeting_notes"

        if any(word in cleaned for word in [
            "chapter",
            "definition",
            "formula",
            "exercise",
            "theorem",
            "example",
        ]):
            return "student_study_material"

        if any(word in cleaned for word in [
            "invoice",
            "amount",
            "payment",
            "total",
            "gst",
        ]):
            return "business_document"

        return "general_document"

    def _looks_like_questions(
        self,
        text: str,
    ) -> bool:
        question_mark_count = text.count("?")

        question_words = [
            "what",
            "why",
            "how",
            "define",
            "explain",
            "describe",
            "calculate",
            "prove",
        ]

        hits = sum(
            1 for word in question_words
            if word in text.lower()
        )

        return question_mark_count >= 2 or hits >= 3

    def _looks_like_meeting_notes(
        self,
        text: str,
    ) -> bool:
        keywords = [
            "agenda",
            "meeting",
            "action item",
            "deadline",
            "decision",
            "follow up",
            "client",
            "project",
        ]

        hits = sum(
            1 for word in keywords
            if word in text.lower()
        )

        return hits >= 2


if __name__ == "__main__":
    system = InputUnderstandingSystem()

    sample_text = """
    Chapter 1: Machine Learning
    Define supervised learning.
    Explain regression and classification with examples.
    """

    result = system.analyze_text(sample_text)

    print(result)