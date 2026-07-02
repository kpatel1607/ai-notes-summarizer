import re
from typing import Any, Dict, List, Tuple


class SemanticChunker:
    """
    General-purpose, context-aware document chunker.

    Main responsibilities:
    - Preserve short documents as complete text.
    - Divide medium and long documents into coherent chunks.
    - Preserve important short fields, list items and structured data.
    - Avoid duplicate chunks from overlapping parser representations.
    - Validate how much of the original document survives structure parsing.
    - Rank chunks only when a genuinely long document must be limited.
    - Restore source order after priority-based selection.

    This class does not determine semantic noise. That should be handled by
    SemanticPreservationFilter before this chunker receives final_text.
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
        "detailed_summary",
        "clean_text",
        "simplify",
        "explain",
        "translate",
    }

    IMPORTANT_SHORT_TERMS = {
        # General information
        "title",
        "summary",
        "overview",
        "date",
        "time",
        "location",
        "contact",
        "warning",
        "instruction",
        "reference",

        # Professional information
        "company",
        "organization",
        "department",
        "role",
        "position",
        "salary",
        "stipend",
        "experience",
        "skills",
        "requirements",
        "responsibilities",
        "qualification",
        "benefits",
        "deadline",
        "apply",
        "email",
        "phone",
        "remote",
        "hybrid",
        "onsite",
        "owner",
        "assignee",
        "decision",
        "action",
        "due date",
        "next step",
        "next steps",
        "risk",
        "budget",
        "status",

        # Student information
        "chapter",
        "topic",
        "objective",
        "definition",
        "concept",
        "formula",
        "theorem",
        "example",
        "question",
        "answer",
        "method",
        "result",
        "conclusion",
        "important note",
        "important notes",
    }

    STRUCTURED_TYPES = {
        "high",
        "rich",
        "table",
        "mixed",
        "key_value",
        "form",
        "list",
        "multi_column",
    }

    def chunk(
        self,
        pipeline_output: Dict[str, Any],
        max_words: int = 350,
        max_chunks: int = 16,
        user_plan: str = "free",
        mode: str = "general",
        task: str = "short_summary",
        structural_richness: str = "unknown",
        short_document_limit: int = 500,
    ) -> Dict[str, Any]:
        """
        Convert processed document text into safe generation chunks.

        Args:
            pipeline_output:
                Expected keys:
                - final_text
                - structure
                - understanding

            max_words:
                Approximate maximum words per chunk.

            max_chunks:
                Base maximum chunk count.

            user_plan:
                free, pro or premium.

            mode:
                general, student or professional.

            task:
                Selected generation task.

            structural_richness:
                Optional structure classification. If unknown, it is inferred.

            short_document_limit:
                Documents at or below this word count are sent as one complete
                chunk because splitting them gives little benefit.

        Returns:
            Dictionary containing chunks and diagnostic metadata.
        """

        text = str(
            pipeline_output.get("final_text", "") or ""
        ).strip()

        structure = pipeline_output.get("structure", {}) or {}
        understanding = pipeline_output.get("understanding", {}) or {}

        mode = (mode or "general").strip().lower()
        task = (task or "short_summary").strip().lower()
        user_plan = (user_plan or "free").strip().lower()
        structural_richness = (
            structural_richness or "unknown"
        ).strip().lower()

        max_words = max(
            50,
            self._safe_int(max_words, default=350),
        )

        max_chunks = max(
            1,
            self._safe_int(max_chunks, default=16),
        )

        short_document_limit = max(
            50,
            self._safe_int(
                short_document_limit,
                default=500,
            ),
        )

        metadata = understanding.get("metadata", {}) or {}

        page_count = self._safe_int(
            metadata.get("page_count", 1),
            default=1,
        )

        if structural_richness in {"", "unknown"}:
            structural_richness = self._infer_structural_richness(
                structure
            )

        word_count = self._count_words(text)

        effective_max_chunks = self._resolve_max_chunks(
            max_chunks=max_chunks,
            user_plan=user_plan,
        )

        if not text:
            return self._empty_result(
                max_words=max_words,
                max_chunks=effective_max_chunks,
                page_count=page_count,
                user_plan=user_plan,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
                short_document_limit=short_document_limit,
            )

        # General rule:
        # A short document that already fits comfortably should not be split,
        # ranked, filtered or reordered.
        if word_count <= short_document_limit:
            return self._full_text_result(
                text=text,
                word_count=word_count,
                max_words=max_words,
                max_chunks=effective_max_chunks,
                page_count=page_count,
                user_plan=user_plan,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
                short_document_limit=short_document_limit,
            )

        layout_sections = structure.get("layout_sections", []) or []
        sections = structure.get("sections", []) or []
        paragraphs = structure.get("paragraphs", []) or []
        numbered_items = structure.get("numbered_items", []) or []
        roman_items = structure.get("roman_items", []) or []
        key_value_fields = structure.get("key_value_fields", []) or []
        tables = structure.get("tables", []) or []

        chunks = self._build_priority_chunks(
            layout_sections=layout_sections,
            numbered_items=numbered_items,
            roman_items=roman_items,
            key_value_fields=key_value_fields,
            tables=tables,
            sections=sections,
            paragraphs=paragraphs,
            text=text,
            max_words=max_words,
            mode=mode,
            task=task,
            structural_richness=structural_richness,
        )

        chunks = self._remove_empty_chunks(
            chunks=chunks,
            mode=mode,
            task=task,
            structural_richness=structural_richness,
        )

        chunks = self._deduplicate_chunks(chunks)

        structured_chunk_word_count = self._total_chunk_words(chunks)

        structure_preservation_ratio = min(
            structured_chunk_word_count / max(word_count, 1),
            1.0,
        )

        minimum_structure_ratio = self._minimum_structure_ratio(
            word_count=word_count,
            mode=mode,
            task=task,
            structural_richness=structural_richness,
        )

        warnings: List[str] = []
        structure_fallback_used = False

        # If parser-generated structures cover too little of the source text,
        # discard them and safely chunk the complete plain text.
        if (
            not chunks
            or structure_preservation_ratio < minimum_structure_ratio
        ):
            warnings.append(
                "structure_fallback: "
                f"preserved {structure_preservation_ratio:.2%}, "
                f"required {minimum_structure_ratio:.2%}"
            )

            chunks = self._chunk_plain_text(
                text=text,
                max_words=max_words,
            )

            chunks = self._deduplicate_chunks(chunks)

            structured_chunk_word_count = self._total_chunk_words(chunks)

            structure_preservation_ratio = min(
                structured_chunk_word_count / max(word_count, 1),
                1.0,
            )

            structure_fallback_used = True

        original_chunk_count = len(chunks)

        true_long_document = (
            page_count > 20
            or word_count > max_words * effective_max_chunks
        )

        limit_required = (
            true_long_document
            and original_chunk_count > effective_max_chunks
        )

        selection_used = False

        if limit_required:
            selection_used = True

            ranked_chunks = self._rank_chunks(chunks)

            selected_chunks = ranked_chunks[
                :effective_max_chunks
            ]

            # Priority decides which chunks survive, but the model still
            # receives those chunks in their original reading order.
            limited_chunks = sorted(
                selected_chunks,
                key=self._source_order_key,
            )
        else:
            limited_chunks = list(chunks)

        limited_chunk_word_count = self._total_chunk_words(
            limited_chunks
        )

        final_preservation_ratio = min(
            limited_chunk_word_count / max(word_count, 1),
            1.0,
        )

        limit_applied = (
            original_chunk_count > len(limited_chunks)
        )

        recommended_strategy = (
            "hierarchical_summary"
            if true_long_document
            else "single_pass_generation"
        )

        return {
            "chunks": limited_chunks,
            "metadata": {
                "chunk_count": len(limited_chunks),
                "original_chunk_count": original_chunk_count,
                "max_words": max_words,
                "max_chunks": effective_max_chunks,
                "limit_applied": limit_applied,
                "chunk_limit_required": limit_required,
                "selection_used": selection_used,
                "long_document": true_long_document,
                "page_count": page_count,
                "word_count": word_count,
                "chunk_word_count": limited_chunk_word_count,
                "preservation_ratio": final_preservation_ratio,
                "structure_preservation_ratio":
                    structure_preservation_ratio,
                "minimum_structure_preservation_ratio":
                    minimum_structure_ratio,
                "structure_fallback_used":
                    structure_fallback_used,
                "short_document_full_text": False,
                "short_document_limit": short_document_limit,
                "user_plan": user_plan,
                "mode": mode,
                "task": task,
                "structural_richness":
                    structural_richness,
                "recommended_strategy":
                    recommended_strategy,
                "warnings": warnings,
            },
        }

    def _build_priority_chunks(
        self,
        layout_sections: List[Dict[str, Any]],
        numbered_items: List[Dict[str, Any]],
        roman_items: List[Any],
        key_value_fields: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        paragraphs: List[str],
        text: str,
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        """
        Build chunks from one primary textual representation, then optionally
        include structured blocks that may contain unique information.

        This avoids blindly combining every parser output, which can duplicate
        large portions of the same source document.
        """

        chunks: List[Dict[str, Any]] = []
        document_order = 0

        def append_chunks(
            incoming: List[Dict[str, Any]],
        ) -> None:
            nonlocal document_order

            for chunk in incoming:
                chunk["document_order"] = document_order
                document_order += 1
                chunks.append(chunk)

        # Choose one primary text representation.
        if layout_sections:
            append_chunks(
                self._chunks_from_layout_sections(
                    layout_sections=layout_sections,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        elif sections:
            append_chunks(
                self._chunks_from_sections(
                    sections=sections,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        elif paragraphs:
            append_chunks(
                self._chunks_from_paragraphs(
                    paragraphs=paragraphs,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        else:
            append_chunks(
                self._chunk_plain_text(
                    text=text,
                    max_words=max_words,
                )
            )

        # Add explicit structured information. Deduplication runs later.
        if key_value_fields:
            append_chunks(
                self._chunks_from_key_values(
                    key_value_fields=key_value_fields,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        if tables:
            append_chunks(
                self._chunks_from_tables(
                    tables=tables,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        if numbered_items:
            append_chunks(
                self._chunks_from_numbered_items(
                    numbered_items=numbered_items,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        if roman_items:
            append_chunks(
                self._chunks_from_roman_items(
                    roman_items=roman_items,
                    max_words=max_words,
                    mode=mode,
                    task=task,
                    structural_richness=structural_richness,
                )
            )

        return chunks

    def _resolve_max_chunks(
        self,
        max_chunks: int,
        user_plan: str,
    ) -> int:
        plan = (user_plan or "free").lower().strip()
        safe_max_chunks = max(1, max_chunks)

        if plan == "premium":
            return max(safe_max_chunks, 40)

        if plan == "pro":
            return max(safe_max_chunks, 20)

        return min(safe_max_chunks, 16)

    def _chunks_from_layout_sections(
        self,
        layout_sections: List[Dict[str, Any]],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, section in enumerate(layout_sections):
            content = str(
                section.get("content", "") or ""
            ).strip()

            section_type = str(
                section.get("section_type", "layout")
                or "layout"
            ).strip()

            if self._is_low_value_content(
                content=content,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                content,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "layout_section",
                        "title": section_type,
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "bbox": section.get("bbox", []),
                        "priority": 80,
                    }
                )

        return chunks

    def _chunks_from_numbered_items(
        self,
        numbered_items: List[Dict[str, Any]],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, item in enumerate(numbered_items):
            title = str(
                item.get("title", "") or ""
            ).strip()

            content = str(
                item.get("text", "") or ""
            ).strip()

            combined = f"{title}\n{content}".strip()

            if self._is_low_value_content(
                content=combined,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                combined,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "numbered_item",
                        "title": title,
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "original_number":
                            item.get("number", ""),
                        "priority":
                            self._score_title_priority(
                                title=title,
                                mode=mode,
                                task=task,
                            ),
                    }
                )

        return chunks

    def _chunks_from_roman_items(
        self,
        roman_items: List[Any],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, item in enumerate(roman_items):
            if isinstance(item, dict):
                content = str(
                    item.get("text")
                    or item.get("content")
                    or ""
                ).strip()

                title = str(
                    item.get("title", "") or ""
                ).strip()
            else:
                content = str(item or "").strip()
                title = ""

            combined = f"{title}\n{content}".strip()

            if self._is_low_value_content(
                content=combined,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                combined,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "roman_item",
                        "title": title,
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "priority": 55,
                    }
                )

        return chunks

    def _chunks_from_key_values(
        self,
        key_value_fields: List[Dict[str, Any]],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, field in enumerate(key_value_fields):
            key = str(
                field.get("key", "") or ""
            ).strip()

            value = str(
                field.get("value", "") or ""
            ).strip()

            if not key and not value:
                continue

            combined = (
                f"{key}: {value}".strip()
                if key
                else value
            )

            if self._is_low_value_key_value(
                key=key,
                value=value,
            ):
                continue

            parts = self._split_by_word_limit(
                combined,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "key_value",
                        "title": key,
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "priority":
                            self._score_title_priority(
                                title=key,
                                mode=mode,
                                task=task,
                            ),
                    }
                )

        return chunks

    def _chunks_from_tables(
        self,
        tables: List[Dict[str, Any]],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, table in enumerate(tables):
            rows = table.get("rows", []) or []

            if not rows:
                continue

            markdown = self._table_to_markdown(rows)

            if self._is_low_value_content(
                content=markdown,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                markdown,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "table",
                        "title": f"Table {index + 1}",
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "priority": 95,
                    }
                )

        return chunks

    def _table_to_markdown(
        self,
        rows: List[List[str]],
    ) -> str:
        if not rows:
            return ""

        width = max(
            (len(row) for row in rows),
            default=0,
        )

        if width <= 0:
            return ""

        normalized_rows: List[List[str]] = []

        for row in rows:
            safe_row = [
                str(cell or "").strip()
                for cell in row
            ]

            safe_row.extend(
                [""] * (width - len(safe_row))
            )

            normalized_rows.append(safe_row)

        header = normalized_rows[0]
        separator = ["---"] * width
        body = normalized_rows[1:]

        return "\n".join(
            [
                "| " + " | ".join(header) + " |",
                "| " + " | ".join(separator) + " |",
                *[
                    "| " + " | ".join(row) + " |"
                    for row in body
                ],
            ]
        )

    def _chunks_from_sections(
        self,
        sections: List[Dict[str, Any]],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, section in enumerate(sections):
            heading = str(
                section.get("heading", "") or ""
            ).strip()

            raw_content = section.get("content", [])

            if isinstance(raw_content, list):
                content = "\n".join(
                    str(item or "").strip()
                    for item in raw_content
                    if str(item or "").strip()
                )
            else:
                content = str(
                    raw_content or ""
                ).strip()

            combined = f"{heading}\n{content}".strip()

            if self._is_low_value_content(
                content=combined,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                combined,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "section",
                        "title": heading,
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "priority":
                            self._score_title_priority(
                                title=heading,
                                mode=mode,
                                task=task,
                            ),
                    }
                )

        return chunks

    def _chunks_from_paragraphs(
        self,
        paragraphs: List[str],
        max_words: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for index, paragraph in enumerate(paragraphs):
            paragraph_text = str(
                paragraph or ""
            ).strip()

            if self._is_low_value_content(
                content=paragraph_text,
                mode=mode,
                task=task,
                structural_richness=structural_richness,
            ):
                continue

            parts = self._split_by_word_limit(
                paragraph_text,
                max_words,
            )

            for part_index, part in enumerate(parts):
                chunks.append(
                    {
                        "chunk_type": "paragraph",
                        "title": "",
                        "content": part,
                        "source_index": index,
                        "part_index": part_index,
                        "priority": 40,
                    }
                )

        return chunks

    def _chunk_plain_text(
        self,
        text: str,
        max_words: int,
    ) -> List[Dict[str, Any]]:
        """
        Sentence-aware plain-text chunking.

        Very long individual sentences are split by word count when needed.
        """

        text = (text or "").strip()

        if not text:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+|\n{2,}",
                text,
            )
            if sentence.strip()
        ]

        if not sentences:
            sentences = [text]

        chunks: List[Dict[str, Any]] = []
        current_sentences: List[str] = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = sentence.split()

            if len(sentence_words) > max_words:
                if current_sentences:
                    chunks.append(
                        self._plain_chunk(
                            content=" ".join(
                                current_sentences
                            ),
                            index=len(chunks),
                        )
                    )

                    current_sentences = []
                    current_word_count = 0

                for part in self._split_by_word_limit(
                    sentence,
                    max_words,
                ):
                    chunks.append(
                        self._plain_chunk(
                            content=part,
                            index=len(chunks),
                        )
                    )

                continue

            if (
                current_sentences
                and current_word_count
                + len(sentence_words)
                > max_words
            ):
                chunks.append(
                    self._plain_chunk(
                        content=" ".join(
                            current_sentences
                        ),
                        index=len(chunks),
                    )
                )

                current_sentences = []
                current_word_count = 0

            current_sentences.append(sentence)
            current_word_count += len(sentence_words)

        if current_sentences:
            chunks.append(
                self._plain_chunk(
                    content=" ".join(
                        current_sentences
                    ),
                    index=len(chunks),
                )
            )

        return chunks

    def _plain_chunk(
        self,
        content: str,
        index: int,
    ) -> Dict[str, Any]:
        return {
            "chunk_type": "plain_text",
            "title": "",
            "content": content.strip(),
            "source_index": index,
            "part_index": 0,
            "document_order": index,
            "priority": 30,
        }

    def _split_by_word_limit(
        self,
        text: str,
        max_words: int,
    ) -> List[str]:
        safe_max_words = max(
            50,
            self._safe_int(
                max_words,
                default=350,
            ),
        )

        words = str(text or "").split()

        if not words:
            return []

        if len(words) <= safe_max_words:
            return [str(text).strip()]

        result: List[str] = []

        for index in range(
            0,
            len(words),
            safe_max_words,
        ):
            chunk = " ".join(
                words[
                    index:index + safe_max_words
                ]
            ).strip()

            if chunk:
                result.append(chunk)

        return result

    def _remove_empty_chunks(
        self,
        chunks: List[Dict[str, Any]],
        mode: str,
        task: str,
        structural_richness: str,
    ) -> List[Dict[str, Any]]:
        preserve_short = (
            mode in {"professional", "student"}
            or task in self.PROFESSIONAL_TASKS
            or task in self.STUDENT_TASKS
            or structural_richness
            in self.STRUCTURED_TYPES
        )

        cleaned: List[Dict[str, Any]] = []

        for chunk in chunks:
            content = str(
                chunk.get("content", "") or ""
            ).strip()

            if not content:
                continue

            word_count = self._count_words(content)

            if word_count >= 4:
                cleaned.append(chunk)
                continue

            if (
                preserve_short
                or self._is_important_short_content(
                    content
                )
            ):
                cleaned.append(chunk)

        return cleaned

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove exact and near-exact duplicate chunk text.

        This does not perform aggressive semantic deduplication because similar
        chunks can still contain different facts.
        """

        result: List[Dict[str, Any]] = []
        seen = set()

        for chunk in chunks:
            content = str(
                chunk.get("content", "") or ""
            ).strip()

            normalized = re.sub(
                r"\s+",
                " ",
                content.lower(),
            ).strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(chunk)

        return result

    def _rank_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            chunks,
            key=lambda chunk: (
                self._safe_int(
                    chunk.get("priority", 0),
                    default=0,
                ),
                self._count_words(
                    str(
                        chunk.get(
                            "content",
                            "",
                        )
                        or ""
                    )
                ),
            ),
            reverse=True,
        )

    def _score_title_priority(
        self,
        title: str,
        mode: str,
        task: str,
    ) -> int:
        lower = (title or "").lower().strip()

        universal_high_terms = {
            "abstract",
            "summary",
            "overview",
            "objective",
            "question",
            "answer",
            "definition",
            "formula",
            "theorem",
            "concept",
            "method",
            "result",
            "conclusion",
            "warning",
            "instruction",
            "reference",
        }

        professional_high_terms = {
            "company",
            "organization",
            "department",
            "role",
            "position",
            "location",
            "salary",
            "stipend",
            "experience",
            "skill",
            "requirement",
            "responsibility",
            "qualification",
            "benefit",
            "deadline",
            "apply",
            "email",
            "phone",
            "contact",
            "action",
            "owner",
            "assignee",
            "decision",
            "due date",
            "next step",
            "risk",
            "budget",
            "status",
        }

        student_high_terms = {
            "chapter",
            "topic",
            "objective",
            "definition",
            "formula",
            "theorem",
            "concept",
            "example",
            "question",
            "answer",
            "method",
            "result",
            "conclusion",
            "important note",
            "revision",
        }

        medium_priority_terms = {
            "certificate",
            "declaration",
            "contents",
            "references",
            "appendix",
            "figures",
            "tables",
        }

        if any(
            term in lower
            for term in universal_high_terms
        ):
            return 90

        if (
            mode == "professional"
            or task in self.PROFESSIONAL_TASKS
        ) and any(
            term in lower
            for term in professional_high_terms
        ):
            return 95

        if (
            mode == "student"
            or task in self.STUDENT_TASKS
        ) and any(
            term in lower
            for term in student_high_terms
        ):
            return 95

        if any(
            term in lower
            for term in medium_priority_terms
        ):
            return 60

        return 50

    def _is_low_value_key_value(
        self,
        key: str,
        value: str,
    ) -> bool:
        combined = f"{key} {value}".strip()

        if not combined:
            return True

        if self._is_important_short_content(
            combined
        ):
            return False

        lower = combined.lower()

        bad_terms = {
            "top-left corner",
            "top-right corner",
            "bottom-left",
            "bottom-right",
            "bottom-center",
            "page number",
        }

        return any(
            term in lower
            for term in bad_terms
        )

    def _is_low_value_content(
        self,
        content: str,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> bool:
        text = str(content or "").strip()

        if not text:
            return True

        if self._is_important_short_content(text):
            return False

        lower = text.lower()

        bad_phrases = {
            "top-left corner",
            "top-right corner",
            "bottom-left",
            "bottom-right",
            "bottom-center",
            "page number",
        }

        if any(
            phrase in lower
            for phrase in bad_phrases
        ):
            return True

        word_count = self._count_words(text)

        preserve_short = (
            mode in {"professional", "student"}
            or task in self.PROFESSIONAL_TASKS
            or task in self.STUDENT_TASKS
            or structural_richness
            in self.STRUCTURED_TYPES
        )

        return (
            word_count < 4
            and not preserve_short
        )

    def _is_important_short_content(
        self,
        content: str,
    ) -> bool:
        text = str(content or "").strip()

        if not text:
            return False

        lower = text.lower()

        if any(
            term in lower
            for term in self.IMPORTANT_SHORT_TERMS
        ):
            return True

        # Key-value field.
        if ":" in text:
            return True

        # Email address.
        if re.search(
            r"[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+"
            r"\.[A-Za-z]{2,}",
            text,
        ):
            return True

        # URL.
        if re.search(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # Indian or generic-looking phone number.
        if re.search(
            r"(?:\+91[-\s]?)?[6-9]\d{9}",
            text,
        ):
            return True

        # Amounts, percentages, durations and measurements.
        if re.search(
            r"₹\s?\d+"
            r"|\b\d+(?:\.\d+)?\s?"
            r"(?:lpa|inr|usd|eur|gbp|"
            r"years?|months?|days?|hours?|"
            r"minutes?|kg|cm|mm|km|%|percent)"
            r"\b",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        # Numbered or bulleted list item.
        if re.match(
            r"^(?:[-•*]|\d+[.)]|[IVX]+[.)])\s+",
            text,
            flags=re.IGNORECASE,
        ):
            return True

        return False

    def _infer_structural_richness(
        self,
        structure: Dict[str, Any],
    ) -> str:
        if structure.get("tables"):
            return "table"

        if structure.get("key_value_fields"):
            return "key_value"

        active_types = sum(
            bool(structure.get(key))
            for key in (
                "layout_sections",
                "sections",
                "paragraphs",
                "numbered_items",
                "roman_items",
                "key_value_fields",
                "tables",
            )
        )

        if active_types >= 3:
            return "mixed"

        if active_types >= 1:
            return "rich"

        return "plain"

    def _minimum_structure_ratio(
        self,
        word_count: int,
        mode: str,
        task: str,
        structural_richness: str,
    ) -> float:
        if word_count <= 1800:
            ratio = 0.82
        elif word_count <= 5000:
            ratio = 0.72
        else:
            ratio = 0.60

        if (
            mode == "professional"
            or task in self.PROFESSIONAL_TASKS
        ):
            ratio = max(
                ratio,
                0.80,
            )

        if (
            mode == "student"
            or task in self.STUDENT_TASKS
        ):
            ratio = max(
                ratio,
                0.78,
            )

        if (
            structural_richness
            in self.STRUCTURED_TYPES
        ):
            ratio = max(
                ratio,
                0.82,
            )

        return ratio

    def _source_order_key(
        self,
        chunk: Dict[str, Any],
    ) -> Tuple[int, int]:
        document_order = self._safe_int(
            chunk.get(
                "document_order",
                chunk.get("source_index", 0),
            ),
            default=0,
        )

        part_index = self._safe_int(
            chunk.get("part_index", 0),
            default=0,
        )

        return document_order, part_index

    def _total_chunk_words(
        self,
        chunks: List[Dict[str, Any]],
    ) -> int:
        return sum(
            self._count_words(
                str(
                    chunk.get(
                        "content",
                        "",
                    )
                    or ""
                )
            )
            for chunk in chunks
        )

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

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        if isinstance(value, bool):
            return int(value)

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        try:
            return int(str(value))
        except (TypeError, ValueError):
            return default

    def _empty_result(
        self,
        max_words: int,
        max_chunks: int,
        page_count: int,
        user_plan: str,
        mode: str,
        task: str,
        structural_richness: str,
        short_document_limit: int,
    ) -> Dict[str, Any]:
        return {
            "chunks": [],
            "metadata": {
                "chunk_count": 0,
                "original_chunk_count": 0,
                "max_words": max_words,
                "max_chunks": max_chunks,
                "limit_applied": False,
                "chunk_limit_required": False,
                "selection_used": False,
                "long_document": False,
                "page_count": page_count,
                "word_count": 0,
                "chunk_word_count": 0,
                "preservation_ratio": 1.0,
                "structure_preservation_ratio": 1.0,
                "minimum_structure_preservation_ratio": 1.0,
                "structure_fallback_used": False,
                "short_document_full_text": False,
                "short_document_limit": short_document_limit,
                "user_plan": user_plan,
                "mode": mode,
                "task": task,
                "structural_richness": structural_richness,
                "recommended_strategy":
                    "single_pass_generation",
                "warnings": ["empty_input"],
            },
        }

    def _full_text_result(
        self,
        text: str,
        word_count: int,
        max_words: int,
        max_chunks: int,
        page_count: int,
        user_plan: str,
        mode: str,
        task: str,
        structural_richness: str,
        short_document_limit: int,
    ) -> Dict[str, Any]:
        return {
            "chunks": [
                {
                    "chunk_type": "full_text",
                    "title": "Full Document",
                    "content": text,
                    "source_index": 0,
                    "part_index": 0,
                    "document_order": 0,
                    "priority": 100,
                }
            ],
            "metadata": {
                "chunk_count": 1,
                "original_chunk_count": 1,
                "max_words": max_words,
                "max_chunks": max_chunks,
                "limit_applied": False,
                "chunk_limit_required": False,
                "selection_used": False,
                "long_document": False,
                "page_count": page_count,
                "word_count": word_count,
                "chunk_word_count": word_count,
                "preservation_ratio": 1.0,
                "structure_preservation_ratio": 1.0,
                "minimum_structure_preservation_ratio": 1.0,
                "structure_fallback_used": False,
                "short_document_full_text": True,
                "short_document_limit": short_document_limit,
                "user_plan": user_plan,
                "mode": mode,
                "task": task,
                "structural_richness": structural_richness,
                "recommended_strategy":
                    "single_pass_generation",
                "warnings": [],
            },
        }