import re
from typing import Dict, Any, List


class SemanticChunker:
    def chunk(
        self,
        pipeline_output: Dict[str, Any],
        max_words: int = 220,
        max_chunks: int = 8,
        user_plan: str = "free",
    ) -> Dict[str, Any]:

        text = pipeline_output.get("final_text", "")
        structure = pipeline_output.get("structure", {})
        understanding = pipeline_output.get("understanding", {})

        metadata = understanding.get("metadata", {})
        page_count = metadata.get("page_count", 1)

        layout_sections = structure.get("layout_sections", [])
        sections = structure.get("sections", [])
        paragraphs = structure.get("paragraphs", [])
        numbered_items = structure.get("numbered_items", [])
        roman_items = structure.get("roman_items", [])
        key_value_fields = structure.get("key_value_fields", [])

        effective_max_chunks = self._resolve_max_chunks(
            max_chunks=max_chunks,
            user_plan=user_plan,
        )

        chunks = self._build_priority_chunks(
            layout_sections=layout_sections,
            numbered_items=numbered_items,
            roman_items=roman_items,
            key_value_fields=key_value_fields,
            sections=sections,
            paragraphs=paragraphs,
            text=text,
            max_words=max_words,
        )

        chunks = self._remove_empty_chunks(chunks)
        chunks = self._rank_chunks(chunks)

        original_chunk_count = len(chunks)
        limited_chunks = chunks[:effective_max_chunks]

        limit_applied = original_chunk_count > len(limited_chunks)

        true_long_document = (
            page_count > 10
            or len(text.split()) > max_words * effective_max_chunks * 2
        )

        chunk_limit_required = limit_applied

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
                "chunk_limit_required": chunk_limit_required,
                "long_document": true_long_document,
                "page_count": page_count,
                "user_plan": user_plan,
                "recommended_strategy": recommended_strategy,
            },
        }

    def _build_priority_chunks(
        self,
        layout_sections: List[Dict[str, Any]],
        numbered_items: List[Dict[str, Any]],
        roman_items: List[Any],
        key_value_fields: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        paragraphs: List[str],
        text: str,
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        if layout_sections:
            chunks.extend(
                self._chunks_from_layout_sections(
                    layout_sections,
                    max_words,
                )
            )

        if numbered_items:
            chunks.extend(
                self._chunks_from_numbered_items(
                    numbered_items,
                    max_words,
                )
            )

        if roman_items:
            chunks.extend(
                self._chunks_from_roman_items(
                    roman_items,
                    max_words,
                )
            )

        if key_value_fields:
            chunks.extend(
                self._chunks_from_key_values(
                    key_value_fields,
                    max_words,
                )
            )

        if sections:
            chunks.extend(
                self._chunks_from_sections(
                    sections,
                    max_words,
                )
            )

        if not chunks and paragraphs:
            chunks.extend(
                self._chunks_from_paragraphs(
                    paragraphs,
                    max_words,
                )
            )

        if not chunks and text:
            chunks.extend(
                self._chunk_plain_text(
                    text,
                    max_words,
                )
            )

        return chunks

    def _resolve_max_chunks(
        self,
        max_chunks: int,
        user_plan: str,
    ) -> int:

        plan = user_plan.lower().strip()

        if plan == "premium":
            return max(max_chunks, 40)

        if plan == "pro":
            return max(max_chunks, 20)

        return min(max_chunks, 8)

    def _chunks_from_layout_sections(
        self,
        layout_sections: List[Dict[str, Any]],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for section in layout_sections:
            content = section.get("content", "")
            section_type = section.get("section_type", "layout")

            if self._is_low_value_content(content):
                continue

            for part in self._split_by_word_limit(content, max_words):
                chunks.append(
                    {
                        "chunk_type": "layout_section",
                        "title": section_type,
                        "content": part,
                        "source_index": section.get("section_id", ""),
                        "bbox": section.get("bbox", []),
                        "priority": 80,
                    }
                )

        return chunks

    def _chunks_from_numbered_items(
        self,
        numbered_items: List[Dict[str, Any]],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for item in numbered_items:
            title = item.get("title", "")
            content = item.get("text", "")
            combined = f"{title}\n{content}".strip()

            if self._is_low_value_content(combined):
                continue

            for part in self._split_by_word_limit(combined, max_words):
                chunks.append(
                    {
                        "chunk_type": "numbered_item",
                        "title": title,
                        "content": part,
                        "source_index": item.get("number", ""),
                        "priority": self._score_title_priority(title),
                    }
                )

        return chunks

    def _chunks_from_roman_items(
        self,
        roman_items: List[Any],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for index, item in enumerate(roman_items):
            if isinstance(item, dict):
                content = item.get("text") or item.get("content") or ""
                title = item.get("title", "")
            else:
                content = str(item)
                title = ""

            if self._is_low_value_content(content):
                continue

            for part in self._split_by_word_limit(content, max_words):
                chunks.append(
                    {
                        "chunk_type": "roman_item",
                        "title": title,
                        "content": part,
                        "source_index": index,
                        "priority": 55,
                    }
                )

        return chunks

    def _chunks_from_key_values(
        self,
        key_value_fields: List[Dict[str, Any]],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for field in key_value_fields:
            key = field.get("key", "")
            value = field.get("value", "")
            combined = f"{key}: {value}".strip()

            if self._is_low_value_key_value(key, value):
                continue

            for part in self._split_by_word_limit(combined, max_words):
                chunks.append(
                    {
                        "chunk_type": "key_value",
                        "title": key,
                        "content": part,
                        "source_index": "",
                        "priority": self._score_title_priority(key),
                    }
                )

        return chunks

    def _chunks_from_sections(
        self,
        sections: List[Dict[str, Any]],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for index, section in enumerate(sections):
            heading = section.get("heading", "")
            content = " ".join(section.get("content", []))
            combined = f"{heading}\n{content}".strip()

            if self._is_low_value_content(combined):
                continue

            for part in self._split_by_word_limit(combined, max_words):
                chunks.append(
                    {
                        "chunk_type": "section",
                        "title": heading,
                        "content": part,
                        "source_index": index,
                        "priority": self._score_title_priority(heading),
                    }
                )

        return chunks

    def _chunks_from_paragraphs(
        self,
        paragraphs: List[str],
        max_words: int,
    ) -> List[Dict[str, Any]]:

        chunks = []

        for index, paragraph in enumerate(paragraphs):
            if self._is_low_value_content(paragraph):
                continue

            for part in self._split_by_word_limit(paragraph, max_words):
                chunks.append(
                    {
                        "chunk_type": "paragraph",
                        "title": "",
                        "content": part,
                        "source_index": index,
                        "priority": 40,
                    }
                )

        return chunks

    def _chunk_plain_text(
        self,
        text: str,
        max_words: int,
    ) -> List[Dict[str, Any]]:

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current = []

        for sentence in sentences:
            current.append(sentence)

            if len(" ".join(current).split()) >= max_words:
                chunks.append(
                    {
                        "chunk_type": "plain_text",
                        "title": "",
                        "content": " ".join(current).strip(),
                        "source_index": len(chunks),
                        "priority": 30,
                    }
                )

                current = []

        if current:
            chunks.append(
                {
                    "chunk_type": "plain_text",
                    "title": "",
                    "content": " ".join(current).strip(),
                    "source_index": len(chunks),
                    "priority": 30,
                }
            )

        return chunks

    def _split_by_word_limit(
        self,
        text: str,
        max_words: int,
    ) -> List[str]:

        words = text.split()

        if len(words) <= max_words:
            return [text.strip()] if text.strip() else []

        chunks = []

        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words]).strip()

            if chunk:
                chunks.append(chunk)

        return chunks

    def _remove_empty_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        cleaned = []

        for chunk in chunks:
            content = chunk.get("content", "").strip()

            if len(content.split()) < 4:
                continue

            cleaned.append(chunk)

        return cleaned

    def _rank_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        return sorted(
            chunks,
            key=lambda chunk: chunk.get("priority", 0),
            reverse=True,
        )

    def _score_title_priority(
        self,
        title: str,
    ) -> int:

        lower = title.lower()

        high_priority_terms = [
            "abstract",
            "summary",
            "objective",
            "chapter",
            "question",
            "definition",
            "concept",
            "method",
            "result",
            "conclusion",
            "requirement",
            "deadline",
            "action",
            "experience",
            "project",
            "skill",
            "education",
        ]

        medium_priority_terms = [
            "certificate",
            "declaration",
            "contents",
            "references",
            "appendix",
            "figures",
            "tables",
        ]

        if any(term in lower for term in high_priority_terms):
            return 90

        if any(term in lower for term in medium_priority_terms):
            return 60

        return 50

    def _is_low_value_key_value(
        self,
        key: str,
        value: str,
    ) -> bool:

        combined = f"{key} {value}".lower()

        bad_terms = [
            "top-left corner",
            "top-right corner",
            "bottom-left",
            "bottom-right",
            "bottom-center",
            "corner",
            "page number",
        ]

        return any(term in combined for term in bad_terms)

    def _is_low_value_content(
        self,
        content: str,
    ) -> bool:

        text = content.strip()

        if not text:
            return True

        lower = text.lower()

        bad_phrases = [
            "top-left corner",
            "top-right corner",
            "bottom-left",
            "bottom-right",
            "bottom-center",
            "page number",
        ]

        if any(phrase in lower for phrase in bad_phrases):
            return True

        if len(text.split()) < 4:
            return True

        return False