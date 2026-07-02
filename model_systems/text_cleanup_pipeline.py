import re
from typing import Any, Dict, List, Tuple


class TextCleanupPipeline:
    """
    Conservative OCR cleanup pipeline.

    Responsibilities:
    - fix high-confidence spacing and punctuation problems;
    - preserve URLs, email addresses, file names, paths, identifiers, and numbers;
    - preserve technical terms, names, lists, and document structure;
    - remove only highly reliable page noise;
    - fall back to the original text if suspicious content loss occurs.

    This class cleans OCR text. It does not summarize, rank, or remove content
    based on semantic importance.
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
        "detailed_summary",
        "clean_text",
        "simplify",
        "explain",
        "translate",
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

    def clean_ocr_text(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
        structural_richness: str = "unknown",
    ) -> Dict[str, Any]:
        mode = (mode or "general").strip().lower()
        task = (task or "short_summary").strip().lower()
        structural_richness = (
            structural_richness or "unknown"
        ).strip().lower()

        original_text = text or ""
        original_length = len(original_text)
        original_word_count = self._count_words(
            original_text,
        )

        cleaned, protected_tokens = (
            self._protect_special_tokens(
                original_text,
            )
        )

        cleanup_steps: List[str] = []
        warnings: List[str] = []

        cleaned, applied = (
            self._normalize_basic_spacing(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._normalize_bullets(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._remove_simple_page_noise(
                cleaned,
                mode=mode,
                task=task,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._fix_punctuation_spacing(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._fix_common_ocr_merges(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._remove_repeated_punctuation(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned, applied = (
            self._normalize_newlines(
                cleaned,
            )
        )
        cleanup_steps.extend(applied)

        cleaned = self._restore_special_tokens(
            cleaned,
            protected_tokens,
        ).strip()

        cleaned_word_count = self._count_words(
            cleaned,
        )

        minimum_preservation_ratio = (
            self._minimum_preservation_ratio(
                original_word_count=original_word_count,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            )
        )

        preservation_ratio = (
            cleaned_word_count
            / max(original_word_count, 1)
            if original_word_count > 0
            else 1.0
        )

        fallback_used = False

        if (
            original_word_count >= 30
            and preservation_ratio
            < minimum_preservation_ratio
        ):
            warnings.append(
                "cleanup_fallback_used: "
                f"preserved {preservation_ratio:.2%}, "
                f"required "
                f"{minimum_preservation_ratio:.2%}"
            )

            cleaned = original_text.strip()
            cleaned_word_count = (
                original_word_count
            )
            preservation_ratio = 1.0
            fallback_used = True
            cleanup_steps.append(
                "cleanup_fallback",
            )

        return {
            "original_text": original_text,
            "cleaned_text": cleaned,
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "original_word_count": (
                original_word_count
            ),
            "cleaned_word_count": (
                cleaned_word_count
            ),
            "preservation_ratio": (
                preservation_ratio
            ),
            "minimum_preservation_ratio": (
                minimum_preservation_ratio
            ),
            "cleanup_applied": (
                cleaned != original_text.strip()
                and not fallback_used
            ),
            "cleanup_fallback_used": (
                fallback_used
            ),
            "cleanup_steps": cleanup_steps,
            "warnings": warnings,
            "mode": mode,
            "task": task,
            "structural_richness": (
                structural_richness
            ),
        }

    # ------------------------------------------------------------------
    # Protected tokens
    # ------------------------------------------------------------------

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
            protected[placeholder] = (
                match.group(0)
            )
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

        for placeholder, original in (
            protected.items()
        ):
            restored = restored.replace(
                placeholder,
                original,
            )

        return restored

    # ------------------------------------------------------------------
    # Conservative cleanup stages
    # ------------------------------------------------------------------

    def _normalize_basic_spacing(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = text.replace(
            "\r\n",
            "\n",
        )
        fixed = fixed.replace(
            "\r",
            "\n",
        )
        fixed = fixed.replace(
            "\t",
            " ",
        )

        fixed = re.sub(
            r"[ ]{2,}",
            " ",
            fixed,
        )

        return fixed, (
            ["basic_spacing"]
            if fixed != text
            else []
        )

    def _normalize_bullets(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"[•●○▪■]+",
            "-",
            text,
        )

        fixed = re.sub(
            r"^\s*[–—]\s+",
            "- ",
            fixed,
            flags=re.MULTILINE,
        )

        return fixed, (
            ["bullet_normalization"]
            if fixed != text
            else []
        )

    def _remove_simple_page_noise(
        self,
        text: str,
        mode: str,
        task: str,
    ) -> Tuple[str, List[str]]:
        lines = text.splitlines()
        cleaned_lines: List[str] = []
        removed_count = 0

        for line in lines:
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            if self._is_page_noise_line(
                stripped,
                mode=mode,
                task=task,
            ):
                removed_count += 1
                continue

            cleaned_lines.append(line)

        cleaned = "\n".join(
            cleaned_lines
        )

        return cleaned, (
            ["page_noise_removal"]
            if removed_count > 0
            else []
        )

    def _is_page_noise_line(
        self,
        line: str,
        mode: str,
        task: str,
    ) -> bool:
        lower = line.lower().strip()

        noise_patterns = [
            r"^page\s*\d+(\s*of\s*\d+)?$",
            r"^\d+\s*/\s*\d+$",
            r"^this page intentionally left blank$",
            r"^blank page$",
        ]

        conservative_labels = [
            r"^confidential$",
            r"^draft$",
            r"^sample copy$",
        ]

        for pattern in noise_patterns:
            if re.fullmatch(
                pattern,
                lower,
                flags=re.IGNORECASE,
            ):
                return True

        preserve_context = (
            mode == "professional"
            or task in self.PROFESSIONAL_TASKS
            or task in self.STUDENT_TASKS
        )

        if not preserve_context:
            for pattern in (
                conservative_labels
            ):
                if re.fullmatch(
                    pattern,
                    lower,
                    flags=re.IGNORECASE,
                ):
                    return True

        return False

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
        """
        Fix only known, high-confidence OCR merges.

        Global lowercase-uppercase splitting is intentionally avoided because
        JavaScript, TypeScript, GitHub, LinkedIn, FastAPI, and MongoDB are valid.
        """
        common_merges = {
            "ofthe": "of the",
            "ofthese": "of these",
            "ofthis": "of this",
            "inthe": "in the",
            "onthe": "on the",
            "tothe": "to the",
            "forthe": "for the",
            "andthe": "and the",
            "withthe": "with the",
            "fromthe": "from the",
            "bythe": "by the",
            "atthe": "at the",
            "wordcell": "word cell",
            "thingsappear": "things appear",
            "separateunits": "separate units",
        }

        fixed = text

        for wrong, correct in (
            common_merges.items()
        ):
            fixed = re.sub(
                rf"\b{re.escape(wrong)}\b",
                correct,
                fixed,
                flags=re.IGNORECASE,
            )

        return fixed, (
            ["safe_ocr_merge_fixes"]
            if fixed != text
            else []
        )

    def _remove_repeated_punctuation(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"!{3,}",
            "!!",
            text,
        )
        fixed = re.sub(
            r"\?{3,}",
            "??",
            fixed,
        )
        fixed = re.sub(
            r"\.{4,}",
            "...",
            fixed,
        )
        fixed = re.sub(
            r",{2,}",
            ",",
            fixed,
        )

        return fixed, (
            ["repeated_punctuation"]
            if fixed != text
            else []
        )

    def _normalize_newlines(
        self,
        text: str,
    ) -> Tuple[str, List[str]]:
        fixed = re.sub(
            r"[ \t]+\n",
            "\n",
            text,
        )

        fixed = re.sub(
            r"[ ]{2,}",
            " ",
            fixed,
        )

        fixed = re.sub(
            r"\n{3,}",
            "\n\n",
            fixed,
        )

        return fixed, (
            ["newline_normalization"]
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
        if original_word_count <= 120:
            base_ratio = 0.98
        elif original_word_count <= 500:
            base_ratio = 0.94
        elif original_word_count <= 1800:
            base_ratio = 0.86
        else:
            base_ratio = 0.75

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
            "table",
            "key_value",
            "mixed",
            "rich",
            "high",
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
