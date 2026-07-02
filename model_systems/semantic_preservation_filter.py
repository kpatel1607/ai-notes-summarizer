import re
from typing import Any, Dict, List, Optional


class SemanticPreservationFilter:
    """
    Rule + vector based semantic preservation filter.

    Purpose:
    - Protect important document lines.
    - Remove only obvious noise.
    - Use embeddings to detect semantic noise, not to delete useful content.
    - Default behavior is KEEP.
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

    NOISE_ANCHORS = [
        "page number",
        "document footer",
        "document header",
        "copyright notice",
        "watermark text",
        "downloaded from website",
        "scan artifact",
        "repeated navigation text",
        "blank page marker",
        "sample copy marker",
    ]

    IMPORTANT_PATTERNS = [
        r"\b(email|e-mail|mail)\b",
        r"\b(phone|mobile|contact|call|whatsapp)\b",
        r"\b(apply|deadline|last date|submit|registration)\b",
        r"\b(role|position|job title|vacancy|opening|hiring)\b",
        r"\b(company|organization|institute|department)\b",
        r"\b(location|remote|hybrid|onsite|work from home)\b",
        r"\b(salary|stipend|ctc|package|pay|compensation)\b",
        r"\b(experience|fresher|internship|qualification)\b",
        r"\b(skill|skills|required|requirement|responsibility|responsibilities)\b",
        r"\b(eligibility|benefits|duration|timing|shift)\b",
        r"\b(date|time|venue|address)\b",
        r"\b(objective|definition|formula|concept|method|result|conclusion)\b",
        r"\b(question|answer|example|note|important|summary)\b",
        r"\b(action|owner|assigned|task|follow up|next step)\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",
        r"https?://\S+|www\.\S+",
        r"₹\s?\d+|\b\d+\s?(lpa|k|rs|inr|usd|month|year|years|months)\b",
    ]

    TECH_TERMS = {
        "python", "java", "javascript", "typescript", "react", "reactjs",
        "nextjs", "nodejs", "flutter", "firebase", "fastapi", "django",
        "flask", "mongodb", "mysql", "postgresql", "git", "github",
        "linkedin", "docker", "kubernetes", "aws", "azure", "gcp",
        "tensorflow", "pytorch", "opencv", "machine learning", "ai",
        "rest api", "ui/ux", "figma",
    }

    OBVIOUS_NOISE_PATTERNS = [
        r"^page\s*\d+(\s*of\s*\d+)?$",
        r"^\d+\s*/\s*\d+$",
        r"^confidential$",
        r"^draft$",
        r"^sample copy$",
        r"^blank page$",
        r"^this page intentionally left blank$",
        r"^downloaded from\b.*",
        r"^copyright\b.*",
        r"^all rights reserved\b.*",
        r"^watermark\b.*",
    ]

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_vectors: bool = True,
    ):
        self.embedding_model_name = embedding_model_name
        self.enable_vectors = enable_vectors
        self._embedding_model = None
        self._noise_embeddings = None
        self._vector_available = False

    def filter_text(
        self,
        text: str,
        mode: str = "",
        task: str = "",
        structural_richness: str = "unknown",
        noise_similarity_threshold: float = 0.82,
    ) -> Dict[str, Any]:

        original_text = text or ""
        mode = (mode or "general").strip().lower()
        task = (task or "short_summary").strip().lower()

        original_word_count = self._count_words(original_text)

        if not original_text.strip():
            return self._result(
                original_text=original_text,
                filtered_text="",
                kept_lines=[],
                removed_lines=[],
                warnings=["empty_input"],
                extra={
                    "vector_used": False,
                    "original_word_count": 0,
                    "filtered_word_count": 0,
                },
            )

        lines = self._split_lines(original_text)
        size_band = self._size_band(original_word_count)

        rules = self._resolve_rules(
            mode=mode,
            task=task,
            size_band=size_band,
            structural_richness=structural_richness,
        )

        vector_ready = self._ensure_vector_model()

        kept_lines: List[str] = []
        removed_lines: List[Dict[str, Any]] = []
        warnings: List[str] = []
        seen_normalized = set()

        vector_scores = self._score_noise_similarity(lines) if vector_ready else {}

        if self.enable_vectors and not vector_ready:
            warnings.append("vector_filter_unavailable_fallback_to_rules")

        for index, raw_line in enumerate(lines):
            line = raw_line.strip()

            if not line:
                if rules["preserve_spacing"]:
                    kept_lines.append("")
                continue

            importance_score = self._importance_score(line, mode, task)
            noise_score = vector_scores.get(index, 0.0)

            if self._is_protected_line(line, mode, task):
                kept_lines.append(raw_line)
                continue

            if rules["remove_duplicates"]:
                line_key = self._normalize_for_duplicate_check(line)

                if line_key in seen_normalized and len(line.split()) <= 12:
                    removed_lines.append(
                        {
                            "line": raw_line,
                            "reason": "duplicate_short_line",
                            "noise_similarity": noise_score,
                            "importance_score": importance_score,
                        }
                    )
                    continue

                seen_normalized.add(line_key)

            if self._is_obvious_noise(line):
                if importance_score >= rules["protection_threshold"]:
                    kept_lines.append(raw_line)
                else:
                    removed_lines.append(
                        {
                            "line": raw_line,
                            "reason": "obvious_noise_regex",
                            "noise_similarity": noise_score,
                            "importance_score": importance_score,
                        }
                    )
                continue

            if (
                vector_ready
                and noise_score >= noise_similarity_threshold
                and importance_score <= 0
                and not rules["preserve_short_lines"]
            ):
                removed_lines.append(
                    {
                        "line": raw_line,
                        "reason": "semantic_noise_vector",
                        "noise_similarity": round(noise_score, 4),
                        "importance_score": importance_score,
                    }
                )
                continue

            if self._looks_like_ocr_fragment(line) and not rules["preserve_short_lines"]:
                removed_lines.append(
                    {
                        "line": raw_line,
                        "reason": "ocr_fragment",
                        "noise_similarity": noise_score,
                        "importance_score": importance_score,
                    }
                )
                continue

            if len(line.split()) < rules["min_words_to_keep"]:
                if importance_score > 0 or rules["preserve_short_lines"]:
                    kept_lines.append(raw_line)
                else:
                    removed_lines.append(
                        {
                            "line": raw_line,
                            "reason": "short_low_value_line",
                            "noise_similarity": noise_score,
                            "importance_score": importance_score,
                        }
                    )
                continue

            kept_lines.append(raw_line)

        filtered_text = self._join_lines(kept_lines)
        filtered_word_count = self._count_words(filtered_text)

        fallback_used = False

        if original_word_count >= 30:
            ratio = filtered_word_count / max(original_word_count, 1)
            min_ratio = rules["minimum_preservation_ratio"]

            if ratio < min_ratio:
                warnings.append(
                    f"semantic_filter_fallback: preserved {ratio:.2f}, required {min_ratio:.2f}"
                )
                filtered_text = original_text.strip()
                filtered_word_count = original_word_count
                removed_lines = []
                fallback_used = True

        return self._result(
            original_text=original_text,
            filtered_text=filtered_text,
            kept_lines=kept_lines,
            removed_lines=removed_lines,
            warnings=warnings,
            extra={
                "mode": mode,
                "task": task,
                "size_band": size_band,
                "structural_richness": structural_richness,
                "rules": rules,
                "vector_used": vector_ready,
                "embedding_model": self.embedding_model_name if vector_ready else None,
                "original_word_count": original_word_count,
                "filtered_word_count": filtered_word_count,
                "fallback_used": fallback_used,
            },
        )

    def _ensure_vector_model(self) -> bool:
        if not self.enable_vectors:
            return False

        if self._vector_available:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._noise_embeddings = self._embedding_model.encode(
                self.NOISE_ANCHORS,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._vector_available = True
            return True

        except Exception:
            self._embedding_model = None
            self._noise_embeddings = None
            self._vector_available = False
            return False

    def _score_noise_similarity(self, lines: List[str]) -> Dict[int, float]:
        if not self._embedding_model or self._noise_embeddings is None:
            return {}

        useful_lines = []
        indexes = []

        for index, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                continue

            useful_lines.append(stripped)
            indexes.append(index)

        if not useful_lines:
            return {}

        try:
            line_embeddings = self._embedding_model.encode(
                useful_lines,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            scores: Dict[int, float] = {}

            for local_index, embedding in enumerate(line_embeddings):
                similarities = self._noise_embeddings @ embedding
                scores[indexes[local_index]] = float(similarities.max())

            return scores

        except Exception:
            return {}

    def _resolve_rules(
        self,
        mode: str,
        task: str,
        size_band: str,
        structural_richness: str,
    ) -> Dict[str, Any]:

        professional = mode == "professional" or task in self.PROFESSIONAL_TASKS
        student = mode == "student" or task in self.STUDENT_TASKS

        rules = {
            "preserve_short_lines": True,
            "preserve_spacing": True,
            "remove_duplicates": True,
            "min_words_to_keep": 1,
            "minimum_preservation_ratio": 0.90,
            "protection_threshold": 1,
        }

        if size_band == "tiny":
            rules.update({
                "minimum_preservation_ratio": 0.98,
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
            })

        elif size_band == "short":
            rules.update({
                "minimum_preservation_ratio": 0.92,
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
            })

        elif size_band == "medium":
            rules.update({
                "minimum_preservation_ratio": 0.80,
                "preserve_short_lines": professional or student,
                "min_words_to_keep": 1 if professional or student else 2,
            })

        else:
            rules.update({
                "minimum_preservation_ratio": 0.65,
                "preserve_short_lines": professional or student,
                "min_words_to_keep": 2 if professional or student else 3,
            })

        if professional:
            rules.update({
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
                "minimum_preservation_ratio": max(
                    rules["minimum_preservation_ratio"],
                    0.90 if size_band in {"tiny", "short"} else 0.78,
                ),
            })

        if student:
            rules.update({
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
                "minimum_preservation_ratio": max(
                    rules["minimum_preservation_ratio"],
                    0.88 if size_band in {"tiny", "short"} else 0.72,
                ),
            })

        if task in {"clean_text", "short_summary"} and size_band == "long":
            rules.update({
                "preserve_short_lines": False,
                "min_words_to_keep": 3,
                "minimum_preservation_ratio": 0.60,
            })

        if task in {
            "table_format",
            "structured_report",
            "email_draft",
            "action_items",
            "meeting_minutes",
        }:
            rules.update({
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
                "minimum_preservation_ratio": 0.90 if size_band in {"tiny", "short"} else 0.78,
            })

        if task in {"flashcards", "mcqs", "qa_generation", "revision_sheet"}:
            rules.update({
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
                "minimum_preservation_ratio": 0.88 if size_band in {"tiny", "short"} else 0.75,
            })

        if structural_richness in {"high", "rich", "table", "mixed"}:
            rules.update({
                "preserve_short_lines": True,
                "min_words_to_keep": 1,
                "minimum_preservation_ratio": max(
                    rules["minimum_preservation_ratio"],
                    0.82,
                ),
            })

        return rules

    def _split_lines(self, text: str) -> List[str]:
        raw_lines = text.replace("\r", "\n").split("\n")
        expanded: List[str] = []

        label_words = (
            "Role|Company|Location|Salary|Stipend|Experience|Skills|"
            "Requirements|Responsibilities|Apply|Email|Phone|Deadline|"
            "Qualification|Benefits|Job Type|Objective|Definition|Formula|"
            "Question|Answer|Action|Owner|Due Date"
        )

        for line in raw_lines:
            stripped = line.strip()

            if not stripped:
                expanded.append("")
                continue

            line = re.sub(
                rf"\s+(?=({label_words})\s*:)",
                "\n",
                line,
                flags=re.IGNORECASE,
            )

            expanded.extend(line.split("\n"))

        return expanded

    def _is_protected_line(self, line: str, mode: str, task: str) -> bool:
        if self._contains_important_pattern(line):
            return True

        lower = line.lower()

        if any(term in lower for term in self.TECH_TERMS):
            return True

        if ":" in line and len(line.split(":")[0].split()) <= 5:
            return True

        if mode == "professional" or task in self.PROFESSIONAL_TASKS:
            if len(line.split()) <= 10:
                job_terms = [
                    "remote", "hybrid", "onsite", "fresher", "intern",
                    "full-time", "part-time", "apply now", "urgent",
                    "immediate", "walk-in",
                ]
                if any(term in lower for term in job_terms):
                    return True

        if mode == "student" or task in self.STUDENT_TASKS:
            if re.match(r"^\d+[\).]\s+", line):
                return True
            if re.match(r"^[A-Z][A-Za-z ]{1,40}:$", line):
                return True

        return False

    def _importance_score(self, line: str, mode: str, task: str) -> int:
        score = 0

        if self._contains_important_pattern(line):
            score += 3

        lower = line.lower()

        if any(term in lower for term in self.TECH_TERMS):
            score += 2

        if ":" in line:
            score += 2

        if re.search(r"\d", line):
            score += 1

        if mode == "professional" or task in self.PROFESSIONAL_TASKS:
            professional_terms = [
                "role", "company", "location", "salary", "apply",
                "experience", "skills", "requirement", "responsibility",
                "deadline", "email", "phone", "qualification",
            ]
            if any(term in lower for term in professional_terms):
                score += 3

        if mode == "student" or task in self.STUDENT_TASKS:
            student_terms = [
                "definition", "formula", "concept", "example", "question",
                "answer", "important", "note", "chapter", "topic",
            ]
            if any(term in lower for term in student_terms):
                score += 3

        return score

    def _contains_important_pattern(self, line: str) -> bool:
        return any(
            re.search(pattern, line, flags=re.IGNORECASE)
            for pattern in self.IMPORTANT_PATTERNS
        )

    def _is_obvious_noise(self, line: str) -> bool:
        lower = line.lower().strip()

        return any(
            re.match(pattern, lower, flags=re.IGNORECASE)
            for pattern in self.OBVIOUS_NOISE_PATTERNS
        )

    def _looks_like_ocr_fragment(self, line: str) -> bool:
        stripped = line.strip()

        if len(stripped) <= 2:
            return True

        if re.fullmatch(r"[^A-Za-z0-9]+", stripped):
            return True

        letters = len(re.findall(r"[A-Za-z]", stripped))
        total = len(stripped)

        return total >= 5 and letters / max(total, 1) < 0.25

    def _size_band(self, word_count: int) -> str:
        if word_count <= 120:
            return "tiny"
        if word_count <= 500:
            return "short"
        if word_count <= 1800:
            return "medium"
        return "long"

    def _join_lines(self, lines: List[str]) -> str:
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _normalize_for_duplicate_check(self, line: str) -> str:
        return re.sub(r"\s+", " ", line.lower().strip())

    def _count_words(self, text: str) -> int:
        return len(re.findall(r"\b\w+\b", text or ""))

    def _result(
        self,
        original_text: str,
        filtered_text: str,
        kept_lines: List[str],
        removed_lines: List[Any],
        warnings: List[str],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        result = {
            "original_text": original_text,
            "filtered_text": filtered_text,
            "kept_lines": kept_lines,
            "removed_lines": removed_lines,
            "semantic_filter_used": True,
            "warnings": warnings,
        }

        if extra:
            result.update(extra)

        return result