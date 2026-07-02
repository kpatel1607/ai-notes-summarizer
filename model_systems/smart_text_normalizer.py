import re
from typing import Any, Dict, List, Tuple


class SmartTextNormalizer:
    """
    Conservative normalizer for OCR and extracted document text.

    It preserves content and machine-readable tokens while reconstructing only
    high-confidence structural boundaries. It is designed to work across
    student, professional, and general tasks.
    """

    PROFESSIONAL_TASKS = {
        "executive_summary",
        "main_points",
        "action_items",
        "meeting_minutes",
        "structured_report",
        "table_format",
        "email_draft",
    }

    STUDENT_TASKS = {
        "important_notes",
        "qa_generation",
        "answer_questions",
        "flashcards",
        "mcqs",
        "beginner_explanation",
        "revision_sheet",
    }

    GENERAL_TASKS = {
        "short_summary",
        "bullet_summary",
        "key_points",
        "clean_text",
        "simplify",
        "explain",
        "translate",
    }

    PROFESSIONAL_MARKERS = {
        "COMPANY",
        "ORGANIZATION",
        "DEPARTMENT",
        "ROLE",
        "POSITION",
        "LOCATION",
        "SALARY",
        "STIPEND",
        "EXPERIENCE",
        "SKILLS",
        "REQUIREMENTS",
        "RESPONSIBILITIES",
        "QUALIFICATIONS",
        "BENEFITS",
        "DEADLINE",
        "APPLY",
        "CONTACT",
        "EMAIL",
        "PHONE",
        "JOB TYPE",
        "ACTION",
        "OWNER",
        "ASSIGNEE",
        "DUE DATE",
        "DECISION",
        "NEXT STEP",
        "NEXT STEPS",
        "AGENDA",
        "DISCUSSION",
        "OUTCOME",
        "RISK",
        "BUDGET",
        "STATUS",
    }

    STUDENT_MARKERS = {
        "PAPER",
        "MARGINS",
        "CONTENTS",
        "NUMBERING",
        "CHAPTER",
        "TOPIC",
        "OBJECTIVE",
        "DEFINITION",
        "CONCEPT",
        "FORMULA",
        "THEOREM",
        "METHOD",
        "EXAMPLE",
        "QUESTION",
        "ANSWER",
        "RESULT",
        "CONCLUSION",
        "IMPORTANT NOTE",
        "IMPORTANT NOTES",
        "FIGURES",
        "TABLES",
        "REFERENCES",
    }

    GENERAL_MARKERS = {
        "TITLE",
        "SUMMARY",
        "INTRODUCTION",
        "BACKGROUND",
        "OVERVIEW",
        "DETAILS",
        "INSTRUCTIONS",
        "WARNING",
        "NOTE",
        "DATE",
        "TIME",
        "LOCATION",
        "CONTACT",
        "CONCLUSION",
        "REFERENCE",
        "REFERENCES",
    }

    PROTECTED_TOKEN_PATTERN = re.compile(
        r"""
        https?://[^\s<>"']+
        |
        www\.[^\s<>"']+
        |
        \b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b
        |
        \b[A-Za-z0-9_-]+\.(?:pdf|docx?|xlsx?|pptx?|csv|json|txt|zip)\b
        |
        \b(?:[A-Za-z]:\\|/)[^\s<>"']+
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def normalize(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
        structural_richness: str = "unknown",
    ) -> Dict[str, Any]:
        original_text = text or ""
        normalized = original_text

        mode = (mode or "general").strip().lower()
        task = (task or "short_summary").strip().lower()
        structural_richness = (
            structural_richness or "unknown"
        ).strip().lower()

        original_word_count = self._count_words(
            original_text,
        )
        applied_fixes: List[str] = []
        warnings: List[str] = []

        normalized, applied = self._repair_split_machine_tokens(
            normalized,
        )
        applied_fixes.extend(applied)

        normalized, protected_tokens = (
            self._protect_special_tokens(
                normalized,
            )
        )

        for operation in (
            self._fix_punctuation_spacing,
            self._fix_common_ocr_merges,
            self._normalize_number_spacing,
            self._split_question_style_headings,
            self._normalize_inline_list_separators,
            self._split_instruction_boundaries,
        ):
            normalized, applied = operation(
                normalized,
            )
            applied_fixes.extend(applied)

        normalized, applied = (
            self._split_dense_document_markers(
                normalized,
                mode=mode,
                task=task,
            )
        )
        applied_fixes.extend(applied)

        normalized, applied = (
            self._fix_roman_list_spacing(
                normalized,
            )
        )
        applied_fixes.extend(applied)

        normalized, applied = (
            self._fix_repeated_spaces(
                normalized,
            )
        )
        applied_fixes.extend(applied)

        normalized = self._restore_special_tokens(
            normalized,
            protected_tokens,
        ).strip()

        normalized_word_count = self._count_words(
            normalized,
        )

        minimum_ratio = self._minimum_preservation_ratio(
            original_word_count=original_word_count,
            mode=mode,
            task=task,
            structural_richness=structural_richness,
        )

        preservation_ratio = (
            normalized_word_count
            / max(original_word_count, 1)
            if original_word_count > 0
            else 1.0
        )

        original_line_count = max(
            1,
            sum(
                1
                for line in original_text.splitlines()
                if line.strip()
            ),
        )
        normalized_line_count = max(
            1,
            sum(
                1
                for line in normalized.splitlines()
                if line.strip()
            ),
        )

        if normalized_line_count > max(
            40,
            original_line_count * 8,
        ):
            warnings.append(
                "normalization_created_many_boundaries: "
                f"{original_line_count} -> "
                f"{normalized_line_count} lines"
            )

        fallback_used = False

        if (
            original_word_count >= 30
            and preservation_ratio < minimum_ratio
        ):
            warnings.append(
                "normalization_fallback_used: "
                f"preserved {preservation_ratio:.2%}, "
                f"required {minimum_ratio:.2%}"
            )

            normalized = original_text.strip()
            normalized_word_count = original_word_count
            preservation_ratio = 1.0
            normalized_line_count = original_line_count
            fallback_used = True
            applied_fixes.append(
                "normalization_fallback",
            )

        return {
            "original_text": original_text,
            "normalized_text": normalized,
            "normalization_applied": (
                normalized != original_text.strip()
                and not fallback_used
            ),
            "applied_fixes": applied_fixes,
            "original_length": len(original_text),
            "normalized_length": len(normalized),
            "original_word_count": original_word_count,
            "normalized_word_count": normalized_word_count,
            "preservation_ratio": preservation_ratio,
            "minimum_preservation_ratio": minimum_ratio,
            "normalization_fallback_used": fallback_used,
            "warnings": warnings,
            "mode": mode,
            "task": task,
            "structural_richness": structural_richness,
            "original_line_count": original_line_count,
            "normalized_line_count": normalized_line_count,
        }

    # ------------------------------------------------------------------
    # Machine-readable token handling
    # ------------------------------------------------------------------

    def _repair_split_machine_tokens(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        """
        Repair only high-confidence OCR splits inside URL path identifiers.

        The second fragment must contain an uppercase letter, digit, underscore,
        or hyphen. This avoids joining ordinary words after a valid URL.
        """
        fixed = re.sub(
            r"""
            (
                https?://
                [A-Za-z0-9.-]+
                /
                [A-Za-z0-9_-]{2,}
            )
            [ \t]+
            (
                (?=[A-Za-z0-9_-]*[A-Z0-9_-])
                [A-Za-z0-9_-]{2,}
            )
            (?=
                [\s,.;:!?#)]
                |
                $
            )
            """,
            r"\1\2",
            text,
            flags=re.VERBOSE,
        )

        return fixed, (
            ["split_machine_token_repair"]
            if fixed != text
            else []
        )

    def _protect_special_tokens(
        self,
        text: str,
    ) -> Tuple[str, Dict[str, str]]:
        protected: Dict[str, str] = {}

        def replace_token(
            match: re.Match,
        ) -> str:
            placeholder = (
                f"LUMINAPROTECTEDTOKEN"
                f"{len(protected)}END"
            )
            protected[placeholder] = match.group(0)
            return placeholder

        return (
            self.PROTECTED_TOKEN_PATTERN.sub(
                replace_token,
                text,
            ),
            protected,
        )

    def _restore_special_tokens(
        self,
        text: str,
        protected: Dict[str, str],
    ) -> str:
        restored = text

        for placeholder, original in protected.items():
            restored = restored.replace(
                placeholder,
                original,
            )

        return restored

    # ------------------------------------------------------------------
    # Conservative normalization
    # ------------------------------------------------------------------

    def _fix_punctuation_spacing(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
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
    ) -> Tuple[str, List[str]]:
        fixes = {
            r"\bofa\b": "of a",
            r"\bofthe\b": "of the",
            r"\bofmany\b": "of many",
            r"\bofseparate\b": "of separate",
            r"\bofthese\b": "of these",
            r"\bofthis\b": "of this",
            r"\binthe\b": "in the",
            r"\bonthe\b": "on the",
            r"\btothe\b": "to the",
            r"\bforthe\b": "for the",
            r"\bandthe\b": "and the",
            r"\bwiththe\b": "with the",
            r"\bfromthe\b": "from the",
            r"\bbythe\b": "by the",
            r"\batthe\b": "at the",
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

    def _normalize_number_spacing(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"\b(\d{1,2})([A-Z][A-Za-z]{2,})",
            r"\1 \2",
            text,
        )

        return fixed, (
            ["safe_number_spacing"]
            if fixed != text
            else []
        )

    def _split_question_style_headings(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        """
        Split short question-shaped headings only when followed by list content.
        """
        fixed = re.sub(
            r"""
            (?<!^)
            [ \t]+
            (
                (?:
                    What(?:'s|\s+is)?
                    |
                    Who
                    |
                    Why
                    |
                    How
                    |
                    Where
                    |
                    When
                    |
                    Which
                )
                [^?\n]{1,55}
                \?
            )
            [ \t]*
            (?=
                [-•●▪■–—]
                [ \t]*
            )
            """,
            r"\n\1\n",
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        return fixed, (
            ["question_style_heading_split"]
            if fixed != text
            else []
        )

    def _normalize_inline_list_separators(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        """
        Reconstruct likely inline lists without damaging numeric ranges,
        phone numbers, identifiers, or ordinary hyphenated words.
        """
        fixed = text

        fixed = re.sub(
            r"(?<![\w/])\s*[•●▪■]\s*",
            "\n- ",
            fixed,
        )

        fixed = re.sub(
            r"(?<![\d/])[ \t]+[-–—][ \t]+(?=[A-Z][A-Za-z])",
            "\n- ",
            fixed,
        )

        fixed = re.sub(
            r"(?<=[A-Za-z)])[ \t]*[-–—][ \t]*"
            r"(?=\d{1,2}[ \t]+[A-Za-z]{2,})",
            "\n- ",
            fixed,
        )

        fixed = re.sub(
            r"(?m)(?:^|(?<=\n)|(?<=[.!?;:]))"
            r"[ \t]*[-–—][ \t]*"
            r"(?=\d{1,2}[ \t]+[A-Za-z]{2,})",
            "\n- ",
            fixed,
        )

        fixed = re.sub(
            r"\n[ \t]{2,}",
            "\n",
            fixed,
        )

        fixed = re.sub(
            r"(?m)^[ \t]*-[ \t]*$\n?",
            "",
            fixed,
        )

        return fixed, (
            ["inline_list_normalization"]
            if fixed != text
            else []
        )

    def _split_instruction_boundaries(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        """
        Split broadly reusable instruction phrases without changing content.
        """
        fixed = re.sub(
            r"""
            (?<!^)
            (?<!\n)
            [ \t]+
            (
                Contact[ \t]+(?:only[ \t]+)?via
                |
                Apply[ \t]+(?:through|via|at)
                |
                Submit[ \t]+(?:through|via|at)
                |
                Register[ \t]+(?:through|via|at)
                |
                Join[ \t]+(?:for|via|at)
                |
                Download[ \t]+(?:from|via)
                |
                Upload[ \t]+(?:through|via|at)
            )
            \b
            """,
            r"\n\1",
            text,
            flags=re.IGNORECASE | re.VERBOSE,
        )

        return fixed, (
            ["instruction_boundary_split"]
            if fixed != text
            else []
        )

    def _split_dense_document_markers(
        self,
        text: str,
        mode: str,
        task: str,
    ) -> Tuple[str, List[str]]:
        """
        Split clear labels and strict numbered structures only.
        """
        fixed = text

        for marker in self._markers_for_context(
            mode=mode,
            task=task,
        ):
            fixed = re.sub(
                rf"(?<!^)(?<!\n)[ \t]+"
                rf"({re.escape(marker)}[ \t]*(?:::|:))",
                r"\n\1",
                fixed,
                flags=re.IGNORECASE,
            )

        fixed = re.sub(
            r"(?m)^[ \t]*([A-Z][A-Z /&()]{2,40})"
            r"[ \t]+[–—-][ \t]+",
            r"\1: ",
            fixed,
        )

        fixed = re.sub(
            r"(?<!^)(?<!\n)[ \t]+"
            r"((?:Q(?:uestion)?[ \t]*\.?[ \t]*"
            r"\d{1,3}[.)]?)[ \t]+)",
            r"\n\1",
            fixed,
            flags=re.IGNORECASE,
        )

        fixed = re.sub(
            r"(?<!^)(?<!\n)[ \t]+"
            r"(\d{1,3}[.)][ \t]+)"
            r"(?=[A-Z]|What\b|Why\b|How\b|Define\b|"
            r"Explain\b|Name\b|State\b|Draw\b)",
            r"\n\1",
            fixed,
            flags=re.IGNORECASE,
        )

        return fixed, (
            ["dense_document_marker_split"]
            if fixed != text
            else []
        )

    def _markers_for_context(
        self,
        mode: str,
        task: str,
    ) -> List[str]:
        if (
            mode == "professional"
            or task in self.PROFESSIONAL_TASKS
        ):
            markers = (
                self.PROFESSIONAL_MARKERS
                | self.GENERAL_MARKERS
            )
        elif (
            mode == "student"
            or task in self.STUDENT_TASKS
        ):
            markers = (
                self.STUDENT_MARKERS
                | self.GENERAL_MARKERS
            )
        else:
            markers = (
                self.PROFESSIONAL_MARKERS
                | self.STUDENT_MARKERS
                | self.GENERAL_MARKERS
            )

        return sorted(
            markers,
            key=len,
            reverse=True,
        )

    def _fix_roman_list_spacing(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"(?<!^)(?<!\n)\s+"
            r"((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)"
            r"[.)]\s+[A-Z])",
            r"\n\1",
            text,
        )

        return fixed, (
            ["roman_list_spacing"]
            if fixed != text
            else []
        )

    def _fix_repeated_spaces(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"[ \t]{2,}",
            " ",
            text,
        )

        fixed = re.sub(
            r"[ \t]+\n",
            "\n",
            fixed,
        )

        fixed = re.sub(
            r"(?m)^[ \t]*-[ \t]*$\n?",
            "",
            fixed,
        )

        fixed = re.sub(
            r"\n[ \t]*\n(?=[ \t]*- )",
            "\n",
            fixed,
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

    # ------------------------------------------------------------------
    # Preservation safety
    # ------------------------------------------------------------------

    def _minimum_preservation_ratio(
        self,
        original_word_count: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> float:
        if original_word_count <= 150:
            base_ratio = 0.98
        elif original_word_count <= 500:
            base_ratio = 0.95
        elif original_word_count <= 1800:
            base_ratio = 0.88
        elif original_word_count <= 5000:
            base_ratio = 0.80
        else:
            base_ratio = 0.72

        if (
            mode == "professional"
            or task in self.PROFESSIONAL_TASKS
        ):
            base_ratio = max(
                base_ratio,
                0.90,
            )
        elif (
            mode == "student"
            or task in self.STUDENT_TASKS
        ):
            base_ratio = max(
                base_ratio,
                0.88,
            )

        if structural_richness in {
            "high",
            "rich",
            "table",
            "mixed",
            "key_value",
            "form",
            "list",
            "multi_column",
        }:
            base_ratio = max(
                base_ratio,
                0.90,
            )

        return base_ratio

    def _count_words(
        self,
        text: str,
    ) -> int:
        return len(
            re.findall(
                r"\b[\w@.+₹$%/-]+\b",
                text or "",
                flags=re.UNICODE,
            )
        )
