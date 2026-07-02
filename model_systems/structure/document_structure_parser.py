import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

try:
    from paddleocr import PPStructure
except ImportError:
    PPStructure = None


class DocumentStructureParser:
    """
    General-purpose structure parser for OCR and extracted document text.

    The parser is intentionally conservative: it detects structure without
    rewriting document meaning or applying document-specific assumptions.
    """

    NUMBERED_ITEM_PATTERN = re.compile(
        r"^\s*(\d{1,3})[.)]\s+(.+?)\s*$"
    )
    ROMAN_ITEM_PATTERN = re.compile(
        r"^\s*((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII))[.)]\s+(.+?)\s*$",
        flags=re.IGNORECASE,
    )
    BULLET_PATTERN = re.compile(
        r"^\s*[-•●▪■–—*]\s+(.+?)\s*$"
    )
    URL_EMAIL_PATTERN = re.compile(
        r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        flags=re.IGNORECASE,
    )
    QUESTION_STARTERS = (
        "what ", "why ", "when ", "where ", "who ", "whom ",
        "whose ", "which ", "how ", "define ", "explain ",
        "describe ", "calculate ", "prove ", "state ", "name ",
        "discuss ", "list ", "compare ", "differentiate ", "draw ",
    )

    KEY_VALUE_LINE_PATTERN = re.compile(
        r"""
        ^\s*
        ([A-Za-z][A-Za-z0-9 /&()_.\-]{1,60})
        \s*
        (?:::|:)
        \s*
        (.+?)
        \s*$
        """,
        flags=re.VERBOSE,
    )

    TITLE_TERMS = {
        "announcement",
        "application",
        "agreement",
        "contract",
        "exam",
        "guideline",
        "guidelines",
        "hiring",
        "invoice",
        "minutes",
        "notice",
        "policy",
        "receipt",
        "report",
        "schedule",
        "syllabus",
        "timetable",
    }

    def __init__(self):
        self.pp_structure = None
        self.section_classifier = None
        self.enable_structure_model = os.getenv(
            "LUMINA_ENABLE_STRUCTURE_MODEL",
            "false",
        ).lower().strip() == "true"

        if PPStructure is not None:
            try:
                self.pp_structure = PPStructure(
                    show_log=False,
                    lang="en",
                )
            except Exception as e:
                print(f"PPStructure initialization failed: {e}")
                self.pp_structure = None

        if self.enable_structure_model:
            try:
                from transformers import pipeline

                model_name = os.getenv(
                    "LUMINA_STRUCTURE_MODEL",
                    "typeform/distilbert-base-uncased-mnli",
                )

                self.section_classifier = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                )
            except Exception as e:
                print(f"Structure classifier initialization failed: {e}")
                self.section_classifier = None


    def _is_valid_heading(
        self,
        line: str,
    ) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        if len(stripped) > 120:
            return False

        if len(stripped.split()) > 14:
            return False

        if self.URL_EMAIL_PATTERN.search(
            stripped,
        ):
            return False

        if "|" in stripped and stripped.count("|") >= 2:
            return False

        if "\t" in stripped:
            return False

        if self.BULLET_PATTERN.match(
            stripped,
        ):
            return False

        if self.NUMBERED_ITEM_PATTERN.match(
            stripped,
        ):
            return False

        if self.ROMAN_ITEM_PATTERN.match(
            stripped,
        ):
            return False

        if self.KEY_VALUE_LINE_PATTERN.match(
            stripped,
        ):
            return False

        if stripped.endswith(
            (".", ",", ";")
        ):
            return False

        if re.search(
            r"\d+\.\s*\d+",
            stripped,
        ):
            return False

        alpha_words = re.findall(
            r"[A-Za-z]+",
            stripped,
        )

        uppercase_words = sum(
            1
            for word in alpha_words
            if word.isupper()
            and len(word) > 1
        )

        lower = stripped.lower()

        has_title_term = any(
            term in lower
            for term in self.TITLE_TERMS
        )

        return bool(
            stripped.endswith(":")
            or stripped.isupper()
            or stripped.istitle()
            or uppercase_words >= 1
            or has_title_term
            or re.match(
                r"^(?:Chapter|Unit|Section)\s+\d+",
                stripped,
                flags=re.IGNORECASE,
            )
        )

    def remove_repeated_page_noise(
        self,
        text: str,
        min_repeat: int = 2,
    ) -> Dict[str, Any]:
        if not text:
            return {
                "cleaned_text": "",
                "removed_noise": [],
                "noise_removed": False,
            }

        pages = re.split(
            r"\f|\n\s*---\s*page\s*\d+\s*---\s*\n",
            text,
            flags=re.IGNORECASE,
        )

        if len(pages) <= 1:
            return {
                "cleaned_text": text,
                "removed_noise": [],
                "noise_removed": False,
            }

        candidate_lines = {}

        for page in pages:
            lines = [
                line.strip()
                for line in page.splitlines()
                if line.strip()
            ]

            edge_lines = lines[:3] + lines[-3:]

            for line in edge_lines:
                normalized = self._normalize_noise_line(line)

                if self._is_noise_candidate(normalized):
                    candidate_lines[normalized] = (
                        candidate_lines.get(normalized, 0) + 1
                    )

        repeated_noise = [
            line
            for line, count in candidate_lines.items()
            if count >= min_repeat
        ]

        cleaned_text = text

        for noise in repeated_noise:
            cleaned_text = re.sub(
                re.escape(noise),
                "",
                cleaned_text,
                flags=re.IGNORECASE,
            )

        cleaned_text = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned_text,
        ).strip()

        return {
            "cleaned_text": cleaned_text,
            "removed_noise": repeated_noise,
            "noise_removed": bool(repeated_noise),
        }

    def _normalize_noise_line(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line).strip()

        line = re.sub(
            r"\bpage\s+\d+\b",
            "page",
            line,
            flags=re.IGNORECASE,
        )

        line = re.sub(
            r"\b\d+\s*/\s*\d+\b",
            "page",
            line,
        )

        return line

    def _is_noise_candidate(self, line: str) -> bool:
        if not line:
            return False

        if len(line.split()) > 12:
            return False

        noise_keywords = [
            "page",
            "copyright",
            "confidential",
            "draft",
            "www.",
            "http",
        ]

        lower = line.lower()

        if any(keyword in lower for keyword in noise_keywords):
            return True

        if re.fullmatch(r"\d+", line):
            return True

        return False

    def parse(self, text: str) -> Dict[str, Any]:
        noise_result = self.remove_repeated_page_noise(text)

        cleaned = noise_result["cleaned_text"].strip()

        normalized_text = self._normalize_inline_headings(
            cleaned,
        )

        lines = [
            line.strip()
            for line in normalized_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return {
                "parser_type": (
                    "transformer_assisted_text_parser"
                    if self.section_classifier is not None
                    else "rule_based_text_parser"
                ),
                "title": "",
                "sections": [],
                "section_labels": [],
                "questions": [],
                "bullets": [],
                "numbered_items": [],
                "roman_items": [],
                "key_value_fields": [],
                "tables": [],
                "links": [],
                "contact_lines": [],
                "paragraphs": [],
                "layout_blocks": [],
                "structural_signals": {
                    "has_sections": False,
                    "has_questions": False,
                    "has_bullets": False,
                    "has_numbered_items": False,
                    "has_roman_items": False,
                    "has_key_value_fields": False,
                    "has_tables": False,
                    "has_paragraphs": False,
                    "structural_confidence": 0.0,
                },
                "metadata": {
                    "line_count": 0,
                    "section_count": 0,
                    "question_count": 0,
                    "bullet_count": 0,
                    "numbered_item_count": 0,
                    "roman_item_count": 0,
                    "key_value_count": 0,
                    "table_count": 0,
                    "link_count": 0,
                    "contact_line_count": 0,
                    "paragraph_count": 0,
                    "layout_block_count": 0,
                    "noise_removed": noise_result["noise_removed"],
                    "removed_noise_count": len(noise_result["removed_noise"]),
                    "removed_noise": noise_result["removed_noise"][:10],
                    "structural_confidence": 0.0,
                },
            }

        title = self._detect_title(lines)
        sections = self._detect_sections(lines)

        if title:
            normalized_title = re.sub(
                r"[!?]{1,4}$",
                "",
                title,
            ).strip().lower()

            filtered_sections = []

            for section in sections:
                normalized_heading = re.sub(
                    r"[!?]{1,4}$",
                    "",
                    str(
                        section.get(
                            "heading",
                            "",
                        )
                    ),
                ).strip().lower()

                if (
                    normalized_heading == normalized_title
                    and not section.get(
                        "content",
                        [],
                    )
                ):
                    continue

                filtered_sections.append(
                    section
                )

            sections = filtered_sections

        questions = self._detect_questions(lines)

        section_headings = {
            str(section.get("heading", "")).strip().lower()
            for section in sections
            if str(section.get("heading", "")).strip()
        }

        questions = [
            question
            for question in questions
            if question.strip().lower()
            not in section_headings
        ]

        bullets = self._detect_bullets(lines)
        numbered_items = self._detect_numbered_items(normalized_text)
        roman_items = self._detect_roman_items(normalized_text)
        key_value_fields = self._detect_key_value_fields(normalized_text)
        paragraphs = self._detect_paragraphs(normalized_text)
        tables = self._detect_text_tables(normalized_text)
        links = self._detect_links(normalized_text)
        contact_lines = self._detect_contact_lines(lines)
        section_labels = self._classify_sections(sections, paragraphs)
        structural_signals = self._build_structural_signals(
            lines=lines,
            sections=sections,
            questions=questions,
            bullets=bullets,
            numbered_items=numbered_items,
            roman_items=roman_items,
            key_value_fields=key_value_fields,
            tables=tables,
            paragraphs=paragraphs,
        )

        return {
            "parser_type": (
                "transformer_assisted_text_parser"
                if self.section_classifier is not None
                else "rule_based_text_parser"
            ),
            "title": title,
            "sections": sections,
            "section_labels": section_labels,
            "questions": questions,
            "bullets": bullets,
            "numbered_items": numbered_items,
            "roman_items": roman_items,
            "key_value_fields": key_value_fields,
            "tables": tables,
            "links": links,
            "contact_lines": contact_lines,
            "paragraphs": paragraphs,
            "layout_blocks": [],
            "structural_signals": structural_signals,
            "metadata": {
                "line_count": len(lines),
                "section_count": len(sections),
                "question_count": len(questions),
                "bullet_count": len(bullets),
                "numbered_item_count": len(numbered_items),
                "roman_item_count": len(roman_items),
                "key_value_count": len(key_value_fields),
                "table_count": len(tables),
                "link_count": len(links),
                "contact_line_count": len(contact_lines),
                "paragraph_count": len(paragraphs),
                "layout_block_count": 0,
                "noise_removed": noise_result["noise_removed"],
                "removed_noise_count": len(noise_result["removed_noise"]),
                "removed_noise": noise_result["removed_noise"][:10],
                "structural_confidence": structural_signals[
                    "structural_confidence"
                ],
            },
        }

    def parse_document_image(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        if self.pp_structure is None:
            return {
                "parser_type": "ppstructure_unavailable",
                "layout_blocks": [],
                "metadata": {
                    "layout_block_count": 0,
                    "error": "PPStructure is not installed or not initialized.",
                },
            }

        try:
            path = Path(image_path)

            rgb_image_path = path

            try:
                image = Image.open(path)

                if image.mode != "RGB":
                    rgb_image_path = path.with_name(
                        f"{path.stem}_rgb.jpg"
                    )

                    image.convert("RGB").save(
                        rgb_image_path,
                        "JPEG",
                        quality=95,
                    )

            except Exception as e:
                return {
                    "parser_type": "image_preprocess_failed",
                    "layout_blocks": [],
                    "metadata": {
                        "layout_block_count": 0,
                        "error": str(e),
                    },
                }

            result = self.pp_structure(str(rgb_image_path))

            layout_blocks = []

            for block in result:
                block_type = block.get("type", "unknown")
                bbox = block.get("bbox", [])
                raw_res = block.get("res", "")

                extracted_text = self._extract_text_from_ppstructure_res(
                    raw_res,
                )

                layout_blocks.append(
                    {
                        "type": block_type,
                        "bbox": bbox,
                        "text": extracted_text,
                        "raw": raw_res,
                    }
                )

            try:
                if rgb_image_path != path:
                    rgb_image_path.unlink()
            except Exception:
                pass

            return {
                "parser_type": "ppstructure_layout_parser",
                "layout_blocks": layout_blocks,
                "metadata": {
                    "layout_block_count": len(layout_blocks),
                    "layout_types": list(
                        {
                            block["type"]
                            for block in layout_blocks
                        }
                    ),
                },
            }

        except Exception as e:
            return {
                "parser_type": "ppstructure_failed",
                "layout_blocks": [],
                "metadata": {
                    "layout_block_count": 0,
                    "error": str(e),
                },
            }

    def merge_text_and_layout_structure(
        self,
        text_structure: Dict[str, Any],
        layout_structure: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not layout_structure:
            return text_structure

        merged = dict(text_structure)

        merged["layout_blocks"] = layout_structure.get(
            "layout_blocks",
            [],
        )

        merged["layout_sections"] = self.extract_layout_sections(
            layout_structure,
        )

        merged["tables"] = [
            *merged.get("tables", []),
            *self.extract_layout_tables(layout_structure),
        ]

        merged["layout_parser_type"] = layout_structure.get(
            "parser_type",
            "unknown",
        )

        merged["metadata"] = {
            **merged.get("metadata", {}),
            "layout_block_count": layout_structure.get(
                "metadata",
                {},
            ).get("layout_block_count", 0),
            "layout_types": layout_structure.get(
                "metadata",
                {},
            ).get("layout_types", []),
            "table_count": len(merged.get("tables", [])),
        }

        return merged

    def extract_layout_sections(
        self,
        layout_structure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sections = []

        layout_blocks = layout_structure.get(
            "layout_blocks",
            [],
        )

        for index, block in enumerate(layout_blocks):
            text = block.get("text", "").strip()

            if not text:
                continue

            sections.append(
                {
                    "section_id": index,
                    "section_type": block.get(
                        "type",
                        "unknown",
                    ),
                    "bbox": block.get(
                        "bbox",
                        [],
                    ),
                    "content": text,
                    "word_count": len(text.split()),
                }
            )

        return sections

    def extract_layout_tables(
        self,
        layout_structure: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tables = []

        for block in layout_structure.get("layout_blocks", []):
            block_type = str(block.get("type", "")).lower()
            text = block.get("text", "").strip()

            if "table" not in block_type or not text:
                continue

            rows = self._rows_from_tabular_text(text)

            if rows:
                tables.append(
                    {
                        "source": "layout_model",
                        "rows": rows,
                        "bbox": block.get("bbox", []),
                    }
                )

        return tables

    def _extract_text_from_ppstructure_res(
        self,
        raw_res: Any,
    ) -> str:
        extracted_lines = []

        if isinstance(raw_res, str):
            return raw_res.strip()

        if isinstance(raw_res, list):
            for item in raw_res:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("res") or ""

                    if text:
                        extracted_lines.append(str(text))

                elif isinstance(item, str):
                    extracted_lines.append(item)

        if isinstance(raw_res, dict):
            text = raw_res.get("text") or raw_res.get("res") or ""

            if text:
                extracted_lines.append(str(text))

        return " ".join(extracted_lines).strip()


    def _normalize_inline_headings(
        self,
        text: str,
    ) -> str:
        normalized = text

        normalized = re.sub(
            r"\s+((?:Chapter|Unit|Section)\s+\d+[:.\-]?\s+[A-Z][A-Za-z0-9 ,:&()\-]{2,100})",
            r"\n\n\1",
            normalized,
            flags=re.IGNORECASE,
        )

        # Only split genuine numbered items when punctuation follows the number.
        normalized = re.sub(
            r"(?<!\S)(\d{1,3}[.)]\s+)",
            r"\n\1",
            normalized,
        )

        normalized = re.sub(
            r"(?<!\S)((?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)[.)]\s+)",
            r"\n\1",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"(?<!\S)[•●▪■]\s*",
            "\n- ",
            normalized,
        )

        normalized = re.sub(r"\n{3,}", "\n\n", normalized)

        return normalized.strip()


    def _detect_title(
        self,
        lines: List[str],
    ) -> str:
        if not lines:
            return ""

        for index, candidate in enumerate(
            lines[:4]
        ):
            display_candidate = candidate.strip()

            if not display_candidate:
                continue

            candidate_for_validation = re.sub(
                r"[!?]{1,4}$",
                "",
                display_candidate,
            ).strip()

            if not candidate_for_validation:
                continue

            if candidate_for_validation.endswith("?"):
                continue

            if self.KEY_VALUE_LINE_PATTERN.match(
                candidate_for_validation,
            ):
                continue

            if self.URL_EMAIL_PATTERN.search(
                candidate_for_validation,
            ):
                continue

            words = candidate_for_validation.split()

            if not 1 <= len(words) <= 14:
                continue

            # The first line gets a slightly broader title test because OCR
            # often loses font and alignment information.
            if index == 0 and self._looks_like_document_title(
                candidate_for_validation,
            ):
                return candidate_for_validation.rstrip(":")

            if self._is_valid_heading(
                candidate_for_validation,
            ):
                return candidate_for_validation.rstrip(":")

        return ""

    def _looks_like_document_title(
        self,
        text: str,
    ) -> bool:
        stripped = text.strip()

        if not stripped:
            return False

        if stripped.endswith(
            (".", ",", ";")
        ):
            return False

        words = stripped.split()

        if not 1 <= len(words) <= 14:
            return False

        alpha_words = re.findall(
            r"[A-Za-z]+",
            stripped,
        )

        has_uppercase_emphasis = any(
            word.isupper()
            and len(word) > 2
            for word in alpha_words
        )

        has_title_term = any(
            term in stripped.lower()
            for term in self.TITLE_TERMS
        )

        return bool(
            stripped.isupper()
            or stripped.istitle()
            or has_uppercase_emphasis
            or has_title_term
        )

    def _detect_sections(
        self,
        lines: List[str],
    ) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        current_section: Optional[
            Dict[str, Any]
        ] = None

        for line in lines:
            stripped = line.strip()

            is_key_value = bool(
                self.KEY_VALUE_LINE_PATTERN.match(
                    stripped,
                )
            )

            is_heading = (
                not is_key_value
                and self._is_valid_heading(
                    stripped,
                )
            )

            if stripped.endswith("?"):
                lower = stripped.lower()

                is_heading = (
                    not is_key_value
                    and len(stripped.split()) <= 8
                    and not self.NUMBERED_ITEM_PATTERN.match(
                        stripped,
                    )
                    and not self.URL_EMAIL_PATTERN.search(
                        stripped,
                    )
                    and not any(
                        conjunction in lower
                        for conjunction in (
                            " and ",
                            " but ",
                            " while ",
                            " because ",
                        )
                    )
                )

            if is_heading:
                if current_section:
                    sections.append(
                        current_section
                    )

                current_section = {
                    "heading": stripped.rstrip(":"),
                    "content": [],
                }
                continue

            if current_section:
                current_section[
                    "content"
                ].append(stripped)

        if current_section:
            sections.append(
                current_section
            )

        # Remove empty, low-information sections that are probably values or
        # formatting artifacts rather than real headings.
        cleaned_sections: List[
            Dict[str, Any]
        ] = []

        for section in sections:
            heading = str(
                section.get(
                    "heading",
                    "",
                )
            ).strip()

            content = [
                str(item).strip()
                for item in section.get(
                    "content",
                    [],
                )
                if str(item).strip()
            ]

            if not content:
                continue

            cleaned_sections.append(
                {
                    "heading": heading,
                    "content": content,
                }
            )

        return cleaned_sections

    def _detect_questions(
        self,
        lines: List[str],
    ) -> List[str]:
        questions: List[str] = []

        for line in lines:
            if self._is_probable_question_item(line):
                questions.append(line.strip())

        return questions

    def _is_probable_question_item(
        self,
        line: str,
    ) -> bool:
        stripped = line.strip()

        if not stripped:
            return False

        if self.URL_EMAIL_PATTERN.search(
            stripped,
        ):
            return False

        lower = stripped.lower()

        numbered_match = self.NUMBERED_ITEM_PATTERN.match(
            stripped,
        )

        if numbered_match:
            body = numbered_match.group(
                2
            ).strip().lower()

            return (
                body.endswith("?")
                or body.startswith(
                    self.QUESTION_STARTERS
                )
            )

        labelled_match = re.match(
            r"^\s*q(?:uestion)?\.?\s*\d+[.)]?\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )

        if labelled_match:
            body = labelled_match.group(
                1
            ).strip().lower()

            return (
                body.endswith("?")
                or body.startswith(
                    self.QUESTION_STARTERS
                )
            )

        if not stripped.endswith("?"):
            return False

        # Unnumbered question-like lines can be section headings. Keep them as
        # questions only when they look like actual requests rather than short
        # navigation headings.
        word_count = len(
            stripped.split()
        )

        if word_count <= 4 and lower.startswith(
            (
                "who can ",
                "what's ",
                "what is included",
                "what are the benefits",
                "how does it work",
            )
        ):
            return False

        return lower.startswith(
            self.QUESTION_STARTERS
        )

    def _detect_bullets(
        self,
        lines: List[str],
    ) -> List[str]:
        bullets: List[str] = []

        for line in lines:
            match = self.BULLET_PATTERN.match(line)

            if match:
                bullets.append(match.group(1).strip())

        return bullets


    def _detect_numbered_items(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            match = self.NUMBERED_ITEM_PATTERN.match(line)

            if not match:
                continue

            number = match.group(1).strip()
            body = match.group(2).strip()
            title, item_text = self._split_item_title_and_text(body)

            items.append(
                {
                    "number": number,
                    "title": title,
                    "text": item_text,
                }
            )

        return items

    def _split_item_title_and_text(
        self,
        body: str,
    ) -> Tuple[str, str]:
        dash_match = re.match(
            r"^(.{2,80}?)\s+[–—-]\s+(.+)$",
            body,
        )

        if dash_match:
            return (
                dash_match.group(1).strip(),
                dash_match.group(2).strip(),
            )

        sentence_match = re.match(
            r"^(.{2,80}?[.:?])\s+(.+)$",
            body,
        )

        if sentence_match:
            return (
                sentence_match.group(1).strip(),
                sentence_match.group(2).strip(),
            )

        return body[:80].strip(), ""


    def _detect_roman_items(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        items: List[Dict[str, str]] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            match = self.ROMAN_ITEM_PATTERN.match(line)

            if not match:
                continue

            numeral = match.group(1).upper()
            body = match.group(2).strip()
            title, item_text = self._split_item_title_and_text(body)

            items.append(
                {
                    "number": numeral,
                    "title": title,
                    "text": item_text,
                }
            )

        return items


    def _detect_key_value_fields(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        fields: List[Dict[str, str]] = []
        seen: set[Tuple[str, str]] = set()

        line_pattern = self.KEY_VALUE_LINE_PATTERN

        for raw_line in text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            match = line_pattern.match(line)

            if not match:
                continue

            key = self._clean_field_key(match.group(1))
            value = match.group(2).strip()

            if not self._valid_key_value(key, value):
                continue

            pair = (key.lower(), value.lower())

            if pair not in seen:
                seen.add(pair)
                fields.append({"key": key, "value": value})

        inline_pattern = re.compile(
            r"""
            (?<!\w)
            ([A-Za-z][A-Za-z0-9 /&()_.\-]{1,40})
            [ \t]*
            (?:::|:)
            [ \t]*
            (?![-•●▪■–—*])
            ([^\n]+?)
            (?=
                [ \t]+
                [A-Za-z][A-Za-z0-9 /&()_.\-]{1,40}
                [ \t]*
                (?:::|:)
                |
                \n
                |
                \Z
            )
            """,
            flags=re.VERBOSE,
        )

        for match in inline_pattern.finditer(text):
            key = self._clean_field_key(match.group(1))
            value = " ".join(match.group(2).split()).strip()

            if not self._valid_key_value(key, value):
                continue

            pair = (key.lower(), value.lower())

            if pair not in seen:
                seen.add(pair)
                fields.append({"key": key, "value": value})

        return fields

    def _clean_field_key(
        self,
        key: str,
    ) -> str:
        return re.sub(r"\s+", " ", key).strip(" -–—:|")

    def _valid_key_value(
        self,
        key: str,
        value: str,
    ) -> bool:
        if not key or not value:
            return False

        if len(key) > 45:
            return False

        if len(key.split()) > 6:
            return False

        if len(value) > 500:
            return False

        if self.URL_EMAIL_PATTERN.fullmatch(
            key,
        ):
            return False

        first_word = key.split()[0].lower()

        if first_word in {
            "join",
            "click",
            "visit",
            "apply",
            "submit",
            "send",
            "open",
            "use",
            "follow",
            "download",
            "upload",
        }:
            return False

        if re.search(
            r"[.!?]$",
            key,
        ):
            return False

        return True

    def _detect_paragraphs(
        self,
        text: str,
    ) -> List[str]:
        paragraphs = [
            para.strip()
            for para in re.split(r"\n\s*\n", text)
            if para.strip()
        ]

        if len(paragraphs) <= 1:
            sentences = re.split(r"(?<=[.!?])\s+", text)

            paragraphs = [
                sentence.strip()
                for sentence in sentences
                if len(sentence.split()) > 8
            ]

        return paragraphs

    def _detect_links(
        self,
        text: str,
    ) -> List[str]:
        links: List[str] = []
        seen: set[str] = set()

        for match in re.finditer(
            r"https?://[^\s<>()]+|www\.[^\s<>()]+",
            text,
            flags=re.IGNORECASE,
        ):
            value = match.group(
                0
            ).rstrip(
                ".,;:!?)]"
            )

            key = value.lower()

            if key not in seen:
                seen.add(key)
                links.append(value)

        return links

    def _detect_contact_lines(
        self,
        lines: List[str],
    ) -> List[str]:
        contacts: List[str] = []

        for line in lines:
            stripped = line.strip()
            lower = stripped.lower()

            if not stripped:
                continue

            has_contact_signal = any(
                signal in lower
                for signal in (
                    "contact",
                    "email",
                    "phone",
                    "mobile",
                    "whatsapp",
                    "call",
                )
            )

            has_machine_contact = bool(
                re.search(
                    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
                    stripped,
                )
                or re.search(
                    r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)",
                    stripped,
                )
            )

            if has_contact_signal or has_machine_contact:
                contacts.append(stripped)

        return contacts

    def _detect_text_tables(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        tables = []
        current_lines = []

        for line in text.splitlines():
            stripped = line.strip()

            if self._looks_tabular(stripped):
                current_lines.append(stripped)
                continue

            if current_lines:
                rows = self._rows_from_tabular_text("\n".join(current_lines))

                if rows:
                    tables.append(
                        {
                            "source": "text_layout",
                            "rows": rows,
                        }
                    )

                current_lines = []

        if current_lines:
            rows = self._rows_from_tabular_text("\n".join(current_lines))

            if rows:
                tables.append(
                    {
                        "source": "text_layout",
                        "rows": rows,
                    }
                )

        return tables

    def _looks_tabular(
        self,
        line: str,
    ) -> bool:
        if not line:
            return False

        if "|" in line and line.count("|") >= 2:
            return True

        if "\t" in line:
            return True

        if len(re.split(r"\s{2,}", line)) >= 3:
            return True

        return False

    def _rows_from_tabular_text(
        self,
        text: str,
    ) -> List[List[str]]:
        rows = []

        for line in text.splitlines():
            stripped = line.strip().strip("|")

            if not stripped:
                continue

            if re.fullmatch(r"[-:| ]+", stripped):
                continue

            if "|" in stripped:
                cells = [cell.strip() for cell in stripped.split("|")]
            elif "\t" in stripped:
                cells = [cell.strip() for cell in stripped.split("\t")]
            else:
                cells = [cell.strip() for cell in re.split(r"\s{2,}", stripped)]

            cells = [cell for cell in cells if cell]

            if len(cells) >= 2:
                rows.append(cells)

        return rows if len(rows) >= 2 else []


    def _build_structural_signals(
        self,
        *,
        lines: List[str],
        sections: List[Dict[str, Any]],
        questions: List[str],
        bullets: List[str],
        numbered_items: List[Dict[str, str]],
        roman_items: List[Dict[str, str]],
        key_value_fields: List[Dict[str, str]],
        tables: List[Dict[str, Any]],
        paragraphs: List[str],
    ) -> Dict[str, Any]:
        signal_count = sum(
            [
                bool(sections),
                bool(questions),
                bool(bullets),
                bool(numbered_items),
                bool(roman_items),
                bool(key_value_fields),
                bool(tables),
                bool(paragraphs),
            ]
        )

        confidence = min(
            1.0,
            signal_count / 7.0
            + min(len(lines), 20) / 120.0,
        )

        return {
            "has_sections": bool(sections),
            "has_questions": bool(questions),
            "has_bullets": bool(bullets),
            "has_numbered_items": bool(numbered_items),
            "has_roman_items": bool(roman_items),
            "has_key_value_fields": bool(key_value_fields),
            "has_tables": bool(tables),
            "has_paragraphs": bool(paragraphs),
            "structural_confidence": round(confidence, 3),
        }

    def _classify_sections(
        self,
        sections: List[Dict[str, Any]],
        paragraphs: List[str],
    ) -> List[Dict[str, Any]]:
        candidates = [
            "summary",
            "objective",
            "definition",
            "requirement",
            "action item",
            "decision",
            "risk",
            "table data",
            "question answer",
            "example",
            "conclusion",
        ]

        samples = []

        for index, section in enumerate(sections[:8]):
            text = (
                f"{section.get('heading', '')} "
                f"{' '.join(section.get('content', []))}"
            ).strip()

            if text:
                samples.append((index, text[:600]))

        if not samples:
            samples = [
                (index, paragraph[:600])
                for index, paragraph in enumerate(paragraphs[:6])
            ]

        labels = []

        for index, text in samples:
            if self.section_classifier is None:
                label = self._heuristic_section_label(text)
                score = 0.0
            else:
                try:
                    result = self.section_classifier(
                        text,
                        candidate_labels=candidates,
                        multi_label=False,
                    )
                    label = result["labels"][0]
                    score = float(result["scores"][0])
                except Exception:
                    label = self._heuristic_section_label(text)
                    score = 0.0

            labels.append(
                {
                    "source_index": index,
                    "label": label,
                    "score": round(score, 3),
                }
            )

        return labels

    def _heuristic_section_label(
        self,
        text: str,
    ) -> str:
        lower = text.lower()

        if "deadline" in lower or "owner" in lower or "action" in lower:
            return "action item"

        if "risk" in lower or "issue" in lower:
            return "risk"

        if self._is_probable_question_item(text.strip()):
            return "question answer"

        if "conclusion" in lower:
            return "conclusion"

        if "objective" in lower or "purpose" in lower:
            return "objective"

        if "definition" in lower or "means" in lower:
            return "definition"

        if "requirement" in lower or "must" in lower or "shall" in lower:
            return "requirement"

        return "summary"
