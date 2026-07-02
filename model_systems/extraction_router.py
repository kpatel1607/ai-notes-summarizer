from pathlib import Path
from typing import Any, Dict, Optional

from model_systems.complex_document_extractor import ComplexDocumentExtractor
from model_systems.digital_pdf_extractor import DigitalPDFExtractor
from model_systems.image_ocr_extractor import ImageOCRExtractor
from model_systems.scanned_pdf_extractor import ScannedPDFExtractor


class ExtractionRouter:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    def __init__(self):
        self.digital_pdf_extractor = DigitalPDFExtractor()
        self.image_ocr_extractor = ImageOCRExtractor()
        self.scanned_pdf_extractor = ScannedPDFExtractor(self.image_ocr_extractor)
        self.complex_document_extractor = ComplexDocumentExtractor()

    def extract_text(self, text: str) -> Dict[str, Any]:
        cleaned = (text or "").strip()

        return {
            "extracted_text": cleaned,
            "text": cleaned,
            "source": "plain_text",
            "extraction_method": "direct_text",
            "confidence": 1.0,
            "tables": [],
            "table_count": 0,
        }

    def extract_file(
        self,
        file_path: str,
        *,
        route_config: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        path = Path(file_path)
        extension = path.suffix.lower()
        route_config = route_config or {}
        features = features or {}

        if extension == ".pdf":
            return self.extract_pdf(
                file_path,
                route_config=route_config,
                features=features,
                mode=mode,
                task=task,
                user_plan=user_plan,
            )

        if extension in self.IMAGE_EXTENSIONS:
            return self.image_ocr_extractor.extract(file_path)

        return {
            "extracted_text": "",
            "text": "",
            "source": "unsupported",
            "extraction_method": "unsupported",
            "confidence": 0.0,
            "error": f"Unsupported file type: {extension}",
            "tables": [],
            "table_count": 0,
        }

    def extract_pdf(
        self,
        file_path: str,
        *,
        route_config: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        route_config = route_config or {}
        features = features or {}

        if self.complex_document_extractor.should_use_docling(
            route_config=route_config,
            features=features,
            mode=mode,
            task=task,
        ):
            complex_result = self.complex_document_extractor.extract(file_path)

            if complex_result.get("extracted_text") or complex_result.get("markdown"):
                return complex_result

        if self.digital_pdf_extractor.has_selectable_text(file_path):
            return self.digital_pdf_extractor.extract(
                file_path,
                use_pdfplumber_fallback=bool(features.get("has_tables")),
            )

        return self.scanned_pdf_extractor.extract(
            file_path,
            user_plan=user_plan,
        )
