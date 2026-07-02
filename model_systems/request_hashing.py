import hashlib
import os
import re


class RequestHasher:
    def __init__(self):
        self.cache_version = os.getenv(
            "LUMINA_GENERATION_CACHE_VERSION",
            "v2_detail_retention",
        ).strip() or "v2_detail_retention"

    def normalize_text(self, text: str) -> str:
        cleaned = text or ""

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[•▪◦■●]+", "-", cleaned)
        cleaned = re.sub(r"[^\S\n]+", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        cleaned = re.sub(r"\n\s+", "\n", cleaned)
        cleaned = self._remove_common_ocr_noise(cleaned)

        return cleaned.strip().lower()

    def hash_text(self, text: str) -> str:
        normalized = self.normalize_text(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def cache_key(
        self,
        *,
        user_id: str,
        task: str,
        mode: str,
        text: str,
    ) -> str:
        input_hash = self.hash_text(text)
        safe_user = self._safe_key_part(user_id)
        safe_mode = self._safe_key_part(mode)
        safe_task = self._safe_key_part(task)

        safe_version = self._safe_key_part(self.cache_version)
        return f"lumina:exact:{safe_version}:{safe_user}:{safe_mode}:{safe_task}:{input_hash}"

    def _remove_common_ocr_noise(self, text: str) -> str:
        text = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "", text, flags=re.I)
        text = re.sub(r"\b\d+\s*/\s*\d+\b", "", text)
        text = re.sub(r"[_]{3,}", " ", text)
        text = re.sub(r"[-]{4,}", " ", text)
        text = re.sub(r"[~`^]{2,}", " ", text)
        return text

    def _safe_key_part(self, value: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value or "")
        return cleaned[:80] or "unknown"
