import re
from typing import Dict, Any, List


class TextCleanupPipeline:

    def clean_ocr_text(
        self,
        text: str,
    ) -> Dict[str, Any]:

        original_length = len(text or "")

        cleaned = text or ""

        cleanup_steps: List[str] = []

        cleaned = self._normalize_basic_spacing(cleaned)
        cleanup_steps.append("basic_spacing")

        cleaned = self._normalize_bullets(cleaned)
        cleanup_steps.append("bullet_normalization")

        cleaned = self._remove_simple_page_noise(cleaned)
        cleanup_steps.append("page_noise_removal")

        cleaned = self._fix_punctuation_spacing(cleaned)
        cleanup_steps.append("punctuation_spacing")

        cleaned = self._fix_common_ocr_merges(cleaned)
        cleanup_steps.append("ocr_merge_fixes")

        cleaned = self._remove_repeated_punctuation(cleaned)
        cleanup_steps.append("repeated_punctuation")

        cleaned = self._normalize_newlines(cleaned)
        cleanup_steps.append("newline_normalization")

        cleaned = cleaned.strip()

        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "original_length": original_length,
            "cleaned_length": len(cleaned),
            "cleanup_applied": True,
            "cleanup_steps": cleanup_steps,
        }

    def _normalize_basic_spacing(
        self,
        text: str,
    ) -> str:

        text = text.replace("\r", "\n")
        text = text.replace("\t", " ")

        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        return text

    def _normalize_bullets(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"[•●○▪■]+",
            "-",
            text,
        )

        text = re.sub(
            r"^\s*[–—]\s+",
            "- ",
            text,
            flags=re.MULTILINE,
        )

        return text

    def _remove_simple_page_noise(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                cleaned_lines.append(line)
                continue

            if self._is_page_noise_line(stripped):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _is_page_noise_line(
        self,
        line: str,
    ) -> bool:

        lower = line.lower().strip()

        noise_patterns = [
            r"^page\s*\d+(\s*of\s*\d+)?$",
            r"^\d+\s*/\s*\d+$",
            r"^-?\s*\d+\s*-?$",
            r"^confidential$",
            r"^draft$",
            r"^sample copy$",
        ]

        for pattern in noise_patterns:
            if re.match(
                pattern,
                lower,
                flags=re.IGNORECASE,
            ):
                return True

        return False

    def _fix_punctuation_spacing(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"([!?;:])([A-Za-z0-9])",
            r"\1 \2",
            text,
        )

        text = re.sub(
            r"(?<!\d)\.([A-Za-z])",
            r". \1",
            text,
        )

        text = re.sub(
            r",([A-Za-z])",
            r", \1",
            text,
        )

        return text

    def _fix_common_ocr_merges(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"([a-z])([A-Z])",
            r"\1 \2",
            text,
        )

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

        for wrong, correct in common_merges.items():
            text = re.sub(
                rf"\b{wrong}\b",
                correct,
                text,
                flags=re.IGNORECASE,
            )

        text = re.sub(
            r"\b(the)([A-Z][a-z]+)\b",
            r"\1 \2",
            text,
        )

        return text

    def _remove_repeated_punctuation(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"([.,!?]){2,}",
            r"\1",
            text,
        )

        return text

    def _normalize_newlines(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"[ ]{2,}",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text