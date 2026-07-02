from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz


try:
    import pdfplumber
except Exception:  # pragma: no cover - optional layout fallback
    pdfplumber = None


class DigitalPDFExtractor:
    def has_selectable_text(self, file_path: str, min_words: int = 20) -> bool:
        word_count = 0

        with fitz.open(file_path) as document:
            for page in document:
                word_count += len(page.get_text("text").split())

                if word_count >= min_words:
                    return True

        return False

    def extract(
        self,
        file_path: str,
        *,
        use_pdfplumber_fallback: bool = False,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(file_path)

        page_texts: List[str] = []
        tables: List[Any] = []

        with fitz.open(file_path) as document:
            page_count = document.page_count
            pages_to_process = min(page_count, max_pages or page_count)

            for page_index in range(pages_to_process):
                page = document.load_page(page_index)
                page_text = self._extract_page_text(page)

                if page_text:
                    page_texts.append(page_text)

                tables.extend(self._extract_pymupdf_tables(page))

        extracted_text = self._clean_text("\n\n".join(page_texts))
        has_selectable_text = len(extracted_text.split()) >= 20

        if use_pdfplumber_fallback and pdfplumber and (not extracted_text or tables):
            fallback = self._extract_with_pdfplumber(file_path, max_pages=max_pages)

            if len(fallback.get("extracted_text", "")) > len(extracted_text):
                extracted_text = fallback["extracted_text"]

            tables.extend(fallback.get("tables", []))

        return {
            "extracted_text": extracted_text,
            "text": extracted_text,
            "page_count": page_count,
            "has_selectable_text": has_selectable_text,
            "extraction_method": "pymupdf",
            "source": "digital_pdf_pymupdf",
            "confidence": 1.0 if has_selectable_text else 0.35,
            "tables": tables,
            "table_count": len(tables),
            "error": "" if has_selectable_text else "No selectable PDF text detected",
        }

    def _extract_page_text(self, page) -> str:
        try:
            blocks = page.get_text("blocks")
            text_blocks = []

            for block in sorted(blocks, key=lambda item: (item[1], item[0])):
                block_text = str(block[4] or "").strip()

                if block_text:
                    text_blocks.append(block_text)

            if text_blocks:
                return "\n".join(text_blocks)
        except Exception:
            pass

        return page.get_text("text").strip()

    def _extract_pymupdf_tables(self, page) -> List[Any]:
        tables = []

        try:
            if not hasattr(page, "find_tables"):
                return tables

            found_tables = page.find_tables()

            for table in found_tables.tables:
                rows = table.extract()
                cleaned_rows = []

                for row in rows:
                    cleaned_row = [self._clean_text(str(cell or "")) for cell in row]

                    if any(cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if cleaned_rows:
                    tables.append(cleaned_rows)
        except Exception:
            pass

        return tables

    def _extract_with_pdfplumber(
        self,
        file_path: str,
        *,
        max_pages: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not pdfplumber:
            return {
                "extracted_text": "",
                "tables": [],
            }

        page_texts = []
        tables = []

        try:
            with pdfplumber.open(file_path) as document:
                pages = document.pages[: max_pages or len(document.pages)]

                for page in pages:
                    text = page.extract_text(layout=True) or page.extract_text() or ""

                    if text.strip():
                        page_texts.append(text.strip())

                    for table in page.extract_tables() or []:
                        cleaned_table = [
                            [self._clean_text(str(cell or "")) for cell in row]
                            for row in table
                        ]
                        cleaned_table = [row for row in cleaned_table if any(row)]

                        if cleaned_table:
                            tables.append(cleaned_table)
        except Exception as exc:
            return {
                "extracted_text": "",
                "tables": [],
                "error": str(exc),
            }

        return {
            "extracted_text": self._clean_text("\n\n".join(page_texts)),
            "tables": tables,
        }

    def _clean_text(self, text: str) -> str:
        lines = []

        for line in (text or "").replace("\r", "\n").splitlines():
            normalized = " ".join(line.split()).strip()

            if normalized:
                lines.append(normalized)

        return "\n".join(lines).strip()
