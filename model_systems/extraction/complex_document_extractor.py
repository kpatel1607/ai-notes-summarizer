from pathlib import Path
from typing import Any, Dict, Optional


try:
    from docling.document_converter import DocumentConverter
except Exception:  # pragma: no cover - optional heavy dependency
    DocumentConverter = None


class ComplexDocumentExtractor:
    _converter = None

    def should_use_docling(
        self,
        *,
        route_config: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        mode: str = "general",
        task: str = "short_summary",
    ) -> bool:
        route_config = route_config or {}
        features = features or {}

        if route_config.get("path") == "heavy_path":
            return True

        if int(features.get("layout_complexity") or 0) >= 65:
            return True

        if features.get("has_tables") and task in {"table_format", "structured_report"}:
            return True

        return mode == "professional" and task in {"table_format", "structured_report"}

    def extract(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        converter = self._get_converter()

        if not converter:
            return {
                "structured_blocks": [],
                "markdown": "",
                "tables": [],
                "reading_order": [],
                "layoutlmv3_ready": True,
                "extraction_method": "docling_unavailable",
                "source": "docling_unavailable",
                "confidence": 0.0,
                "error": "Docling is not installed",
            }

        try:
            converted = converter.convert(str(path))
            document = converted.document
            markdown = document.export_to_markdown()

            return {
                "structured_blocks": [],
                "markdown": markdown,
                "extracted_text": markdown,
                "text": markdown,
                "tables": [],
                "reading_order": [],
                "layoutlmv3_ready": True,
                "extraction_method": "docling",
                "source": "docling",
                "confidence": 0.9 if markdown.strip() else 0.2,
                "error": "",
            }
        except Exception as exc:
            return {
                "structured_blocks": [],
                "markdown": "",
                "tables": [],
                "reading_order": [],
                "layoutlmv3_ready": True,
                "extraction_method": "docling",
                "source": "docling",
                "confidence": 0.0,
                "error": str(exc),
            }

    def _get_converter(self):
        if DocumentConverter is None:
            return None

        if ComplexDocumentExtractor._converter is None:
            try:
                ComplexDocumentExtractor._converter = DocumentConverter()
            except Exception as exc:
                print("Docling unavailable:", str(exc))
                ComplexDocumentExtractor._converter = None

        return ComplexDocumentExtractor._converter

    def extract_with_layoutlmv3(self, file_path: str) -> Dict[str, Any]:
        return {
            "structured_blocks": [],
            "markdown": "",
            "tables": [],
            "reading_order": [],
            "extraction_method": "layoutlmv3_not_configured",
            "source": "layoutlmv3_not_configured",
            "confidence": 0.0,
            "error": "LayoutLMv3 interface is reserved for future advanced layout routing.",
        }
