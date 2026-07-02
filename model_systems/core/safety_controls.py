import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Callable, Dict, Optional, Tuple


try:
    import fitz
except Exception:  # pragma: no cover - optional dependency during static checks
    fitz = None


class SafetyControls:
    def __init__(self):
        self.user_window_seconds = int(os.getenv("LUMINA_USER_RATE_WINDOW_SECONDS", "60"))
        self.ip_window_seconds = int(os.getenv("LUMINA_IP_RATE_WINDOW_SECONDS", "60"))
        self.block_seconds = int(os.getenv("LUMINA_TEMP_BLOCK_SECONDS", "900"))
        self.max_pdf_pages = int(os.getenv("LUMINA_MAX_PDF_PAGES", "40"))
        self.max_free_pdf_pages = int(os.getenv("LUMINA_MAX_FREE_PDF_PAGES", "12"))
        self.ocr_timeout_seconds = int(os.getenv("LUMINA_OCR_TIMEOUT_SECONDS", "120"))

        self.user_endpoint_limits = {
            "v2_generate": int(os.getenv("LUMINA_LIMIT_GENERATE_USER_PER_MIN", "20")),
            "v2_jobs_generate": int(os.getenv("LUMINA_LIMIT_JOBS_USER_PER_MIN", "20")),
            "v2_generate_file": int(os.getenv("LUMINA_LIMIT_FILE_USER_PER_MIN", "8")),
            "v2_feedback": int(os.getenv("LUMINA_LIMIT_FEEDBACK_USER_PER_MIN", "30")),
        }
        self.ip_endpoint_limits = {
            "v2_generate": int(os.getenv("LUMINA_LIMIT_GENERATE_IP_PER_MIN", "60")),
            "v2_jobs_generate": int(os.getenv("LUMINA_LIMIT_JOBS_IP_PER_MIN", "60")),
            "v2_generate_file": int(os.getenv("LUMINA_LIMIT_FILE_IP_PER_MIN", "20")),
            "v2_feedback": int(os.getenv("LUMINA_LIMIT_FEEDBACK_IP_PER_MIN", "80")),
        }
        self.heavy_user_limit = int(os.getenv("LUMINA_LIMIT_HEAVY_USER_PER_MIN", "4"))
        self.heavy_ip_limit = int(os.getenv("LUMINA_LIMIT_HEAVY_IP_PER_MIN", "12"))
        self.violation_threshold = int(os.getenv("LUMINA_VIOLATION_THRESHOLD", "5"))

        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, str], list[float]] = {}
        self._blocks: Dict[str, float] = {}
        self._violations: Dict[str, list[float]] = {}

    def client_ip(self, request) -> str:
        forwarded_for = request.headers.get("x-forwarded-for", "")

        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        return getattr(request.client, "host", "") or "unknown"

    def enforce_endpoint_limit(self, *, user_id: str, ip: str, endpoint: str) -> None:
        self._raise_if_blocked(user_id=user_id, ip=ip)
        user_limit = self.user_endpoint_limits.get(endpoint, 20)
        ip_limit = self.ip_endpoint_limits.get(endpoint, 60)

        if not self._allow(key=("user", user_id, endpoint), limit=user_limit, window=self.user_window_seconds):
            self.register_violation(user_id=user_id, ip=ip)
            raise SafetyError("Too many requests. Please wait a moment and try again.", "user_rate_limited")

        if not self._allow(key=("ip", ip, endpoint), limit=ip_limit, window=self.ip_window_seconds):
            self.register_violation(user_id=user_id, ip=ip)
            raise SafetyError("Too many requests from this network. Please wait and try again.", "ip_rate_limited")

    def enforce_heavy_limit(
        self,
        *,
        user_id: str,
        ip: str,
        route_metadata: Dict[str, Any],
        allow_queue: bool,
    ) -> None:
        route_config = route_metadata.get("routeConfig", {})
        complexity = route_metadata.get("complexity", {})
        features = route_metadata.get("features", {})
        score = int(complexity.get("score") or 0)
        queue_required = route_config.get("queue_required") is True
        heavy = route_config.get("path") == "heavy_path" or queue_required or score >= 62

        if not heavy:
            return

        if not self._allow(key=("user", user_id, "heavy"), limit=self.heavy_user_limit, window=60):
            self.register_violation(user_id=user_id, ip=ip)
            raise SafetyError("Heavy document processing is temporarily limited. Please try again later.", "heavy_user_limited")

        if not self._allow(key=("ip", ip, "heavy"), limit=self.heavy_ip_limit, window=60):
            self.register_violation(user_id=user_id, ip=ip)
            raise SafetyError("Heavy document processing is temporarily limited on this network.", "heavy_ip_limited")

        suspicious = (
            score >= 92
            or int(features.get("word_count") or 0) > 18000
            or int(features.get("page_count") or 0) > self.max_pdf_pages
        )

        if suspicious and not allow_queue:
            self.register_violation(user_id=user_id, ip=ip)
            raise SafetyError("This document needs background processing. Please use the latest app update and try again.", "heavy_requires_queue")

    def validate_pdf_pages(self, file_path: str, *, user_plan: str = "free") -> int:
        if fitz is None:
            return 0

        with fitz.open(file_path) as document:
            page_count = document.page_count

        limit = self.max_free_pdf_pages if user_plan == "free" else self.max_pdf_pages

        if page_count > limit:
            raise SafetyError(
                f"PDF has too many pages for this plan. Maximum allowed is {limit} pages.",
                "file_page_limit",
            )

        return page_count

    def run_with_timeout(self, func: Callable[[], Any], *, timeout_seconds: Optional[int] = None) -> Any:
        timeout = timeout_seconds or self.ocr_timeout_seconds

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func)

            try:
                return future.result(timeout=timeout)
            except TimeoutError:
                raise SafetyError("Document processing timed out. Please try a smaller or cleaner file.", "processing_timeout")

    def register_violation(self, *, user_id: str, ip: str) -> None:
        now = time.time()

        with self._lock:
            for subject in [f"user:{user_id}", f"ip:{ip}"]:
                history = [
                    timestamp
                    for timestamp in self._violations.get(subject, [])
                    if now - timestamp <= self.block_seconds
                ]
                history.append(now)
                self._violations[subject] = history

                if len(history) >= self.violation_threshold:
                    self._blocks[subject] = now + self.block_seconds

    def _raise_if_blocked(self, *, user_id: str, ip: str) -> None:
        now = time.time()

        with self._lock:
            for subject in [f"user:{user_id}", f"ip:{ip}"]:
                blocked_until = self._blocks.get(subject, 0)

                if blocked_until > now:
                    raise SafetyError("Too many suspicious requests. Please try again later.", "temporarily_blocked")

                if blocked_until:
                    self._blocks.pop(subject, None)

    def _allow(self, *, key: Tuple[str, ...], limit: int, window: int) -> bool:
        now = time.time()

        with self._lock:
            history = [
                timestamp
                for timestamp in self._counters.get(key, [])
                if now - timestamp <= window
            ]

            if len(history) >= limit:
                self._counters[key] = history
                return False

            history.append(now)
            self._counters[key] = history
            return True


class SafetyError(Exception):
    def __init__(self, message: str, error_type: str):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
