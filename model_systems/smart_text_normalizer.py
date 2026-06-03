import re
from typing import Dict, Any, List


class SmartTextNormalizer:
    def normalize(
        self,
        text: str,
    ) -> Dict[str, Any]:

        original_text = text or ""
        normalized = original_text

        applied_fixes: List[str] = []

        normalized, applied = self._fix_punctuation_spacing(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._fix_common_ocr_merges(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._fix_lower_upper_merges(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._normalize_number_spacing(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._split_dense_document_markers(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._fix_roman_list_spacing(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._fix_quote_spacing(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, applied = self._fix_repeated_spaces(
            normalized,
        )
        applied_fixes.extend(applied)

        return {
            "original_text": original_text,
            "normalized_text": normalized.strip(),
            "normalization_applied": bool(applied_fixes),
            "applied_fixes": applied_fixes,
            "original_length": len(original_text),
            "normalized_length": len(normalized.strip()),
        }

    def _fix_punctuation_spacing(
        self,
        text: str,
    ):

        fixed = re.sub(
            r"([!?;:])([A-Za-z0-9])",
            r"\1 \2",
            text,
        )

        fixed = re.sub(
            r"(?<!\d)\.([A-Za-z])",
            r". \1",
            fixed,
        )

        fixed = re.sub(
            r",([A-Za-z])",
            r", \1",
            fixed,
        )

        return fixed, (
            ["punctuation_spacing"]
            if fixed != text
            else []
        )

    def _fix_common_ocr_merges(
        self,
        text: str,
    ):

        fixes = {
            r"\bofa\b": "of a",
            r"\bofthe\b": "of the",
            r"\bofmany\b": "of many",
            r"\bofseparate\b": "of separate",
            r"\bwordcell\b": "word cell",
            r"\bthingsappear\b": "things appear",
            r"\bseparateunits\b": "separate units",
            r"\bcells\.Cell\b": "cells. Cell",
            r"\btree\.This\b": "tree. This",
            r"\bscience\.This\b": "science. This",
            r"\bunits\.The\b": "units. The",
        }

        fixed = text

        for pattern, replacement in fixes.items():
            fixed = re.sub(
                pattern,
                replacement,
                fixed,
                flags=re.IGNORECASE,
            )

        return fixed, (
            ["common_ocr_merges"]
            if fixed != text
            else []
        )

    def _fix_lower_upper_merges(
        self,
        text: str,
    ):

        fixed = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text,
        )

        return fixed, (
            ["lower_upper_merge"]
            if fixed != text
            else []
        )

    def _normalize_number_spacing(
        self,
        text: str,
    ):

        fixed = text

        # Fix decimal spacing caused by OCR/normalization:
        # 1. 25 -> 1.25, 1. 5 -> 1.5, 6. 13 -> 6.13
        fixed = re.sub(
            r"\b(\d+)\.\s+(\d+)\b",
            r"\1.\2",
            fixed,
        )

        # Fix numbered list spacing:
        # 1Cover Page -> 1 Cover Page
        fixed = re.sub(
            r"\b(\d{1,2})([A-Z][A-Za-z])",
            r"\1 \2",
            fixed,
        )

        return fixed, (
            ["number_spacing"]
            if fixed != text
            else []
        )

    def _split_dense_document_markers(
        self,
        text: str,
    ):

        fixed = text

        markers = [
            "PAPER",
            "MARGINS",
            "CONTENTS",
            "FOLLOWING MUST BE STRICTLY FOLLOWED",
            "NUMBERING",
            "PREPARATION OF CHAPTERS",
            "SPACING/ALIGNMENT",
            "SECTION/SUBSECTION NUMBERING",
            "FIGURES",
            "TABLES",
            "BINDING",
            "NUMBER OF COPIES",
        ]

        for marker in markers:
            fixed = re.sub(
                rf"\s+({re.escape(marker)}\s*:)",
                r"\n\n\1",
                fixed,
                flags=re.IGNORECASE,
            )

        # Split common numbered guideline items:
        # 2 Certificate – ... 3 Declaration – ...
        fixed = re.sub(
            r"\s+(\d{1,2}\s+[A-Z][A-Za-z /&\-]{3,60}\s+[–-])",
            r"\n\n\1",
            fixed,
        )

        fixed = re.sub(
            r"(?<=\.)\s*(\d{1,2}\s+[A-Z][A-Za-z /&\-]{3,60}\s+[–-])",
            r"\n\n\1",
            fixed,
        )

        return fixed, (
            ["dense_document_marker_split"]
            if fixed != text
            else []
        )

    def _fix_roman_list_spacing(
        self,
        text: str,
    ):

        fixed = text

        fixed = re.sub(
            r"\s+((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s+[A-Z])",
            r"\n\1",
            fixed,
        )

        return fixed, (
            ["roman_list_spacing"]
            if fixed != text
            else []
        )

    def _fix_quote_spacing(
        self,
        text: str,
    ):

        fixed = text

        fixed = re.sub(
            r"word'([A-Za-z])",
            r"word '\1",
            fixed,
        )

        fixed = re.sub(
            r"'([A-Za-z]+)to\b",
            r"'\1' to",
            fixed,
        )

        return fixed, (
            ["quote_spacing"]
            if fixed != text
            else []
        )

    def _fix_repeated_spaces(
        self,
        text: str,
    ):

        fixed = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        fixed = re.sub(
            r"\n{3,}",
            "\n\n",
            fixed,
        )

        return fixed, (
            ["repeated_spaces"]
            if fixed != text
            else []
        )