import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from PIL import Image

try:
    from paddleocr import PPStructure
except ImportError:
    PPStructure = None


class DocumentStructureParser:
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

    def _is_valid_heading(self, line: str) -> bool:
        line = line.strip()

        if not line:
            return False

        word_count = len(line.split())

        if word_count > 10:
            return False

        if len(line) > 80:
            return False

        if line.endswith((".", ",", ";", ":")):
            return False

        invalid_starts = ["1.", "2.", "3.", "4.", "5."]

        lower = line.lower().strip()

        for start in invalid_starts:
            if lower.startswith(start):
                if any(
                    keyword in lower
                    for keyword in [
                        "table",
                        "figure",
                        "spacing",
                        "paragraph",
                        "alignment",
                    ]
                ):
                    return False

        invalid_keywords = [
            "font",
            "spacing",
            "alignment",
            "margin",
            "margins",
            "numbering",
            "bottom",
            "top",
            "corner",
            "page number",
            "times new roman",
        ]

        invalid_count = sum(
            1
            for keyword in invalid_keywords
            if keyword in lower
        )

        if invalid_count >= 1:
            return False

        if len(re.findall(r"\d+\.\s*\d+", line)) >= 1:
            return False

        return True

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
            "university",
            "college",
            "institute",
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
                "title": "",
                "sections": [],
                "questions": [],
                "bullets": [],
                "numbered_items": [],
                "roman_items": [],
                "key_value_fields": [],
                "paragraphs": [],
                "layout_blocks": [],
                "metadata": {},
            }

        title = self._detect_title(lines)
        sections = self._detect_sections(lines)
        questions = self._detect_questions(lines)
        bullets = self._detect_bullets(lines)
        numbered_items = self._detect_numbered_items(normalized_text)
        roman_items = self._detect_roman_items(normalized_text)
        key_value_fields = self._detect_key_value_fields(normalized_text)
        paragraphs = self._detect_paragraphs(normalized_text)
        tables = self._detect_text_tables(normalized_text)
        section_labels = self._classify_sections(sections, paragraphs)

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
            "paragraphs": paragraphs,
            "layout_blocks": [],
            "metadata": {
                "line_count": len(lines),
                "section_count": len(sections),
                "question_count": len(questions),
                "bullet_count": len(bullets),
                "numbered_item_count": len(numbered_items),
                "roman_item_count": len(roman_items),
                "key_value_count": len(key_value_fields),
                "table_count": len(tables),
                "paragraph_count": len(paragraphs),
                "layout_block_count": 0,
                "noise_removed": noise_result["noise_removed"],
                "removed_noise_count": len(noise_result["removed_noise"]),
                "removed_noise": noise_result["removed_noise"][:10],
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

    def _normalize_inline_headings(self, text: str) -> str:
        normalized = text

        normalized = re.sub(
            r"\s+((?:Chapter|Unit|Section)\s+\d+[:.\-]?\s+[A-Z][A-Za-z0-9 ,:&\-]{4,90})",
            r"\n\n\1",
            normalized,
            flags=re.IGNORECASE,
        )

        normalized = re.sub(
            r"\s+(\d+\.\s+[A-Z][A-Za-z ,:&\-]{8,80})(?=\s+[A-Z])",
            r"\n\n\1",
            normalized,
        )

        return normalized.strip()

    def _detect_title(self, lines: List[str]) -> str:
        if not lines:
            return ""

        first = lines[0].strip()

        word_count = len(first.split())

        if word_count > 12:
            return ""

        if first.endswith((".", "?", "!")):
            return ""

        lowercase_ratio = (
            sum(1 for c in first if c.islower())
            / max(len(first), 1)
        )

        if lowercase_ratio > 0.65:
            return ""

        title_patterns = [
            r"^(Chapter|Unit|Section)\s+\d+",
            r"^[A-Z][A-Z0-9 /:&\-\(\)]{4,80}$",
            r"^[A-Z][A-Za-z0-9 ,:&\-\(\)]{3,80}$",
        ]

        for pattern in title_patterns:
            if re.match(pattern, first):
                return first

        return ""

    def _detect_sections(
        self,
        lines: List[str],
    ) -> List[Dict[str, Any]]:
        sections = []
        current_section = None

        heading_pattern = re.compile(
            r"^((?:Chapter|Unit|Section)\s+\d+[:.\-]?\s+.+|[A-Z][A-Z /&\-]{4,80})$",
            re.IGNORECASE,
        )

        numbered_heading_pattern = re.compile(
            r"^\d+\.\s+[A-Z][A-Za-z ,:&\-]{8,80}$"
        )

        for line in lines:
            is_heading = (
                (
                    bool(heading_pattern.match(line))
                    or bool(numbered_heading_pattern.match(line))
                )
                and self._is_valid_heading(line)
            )

            if is_heading:
                if current_section:
                    sections.append(current_section)

                current_section = {
                    "heading": line,
                    "content": [],
                }

            else:
                if current_section:
                    current_section["content"].append(line)

        if current_section:
            sections.append(current_section)

        return sections

    def _detect_questions(
        self,
        lines: List[str],
    ) -> List[str]:
        questions = []

        question_pattern = re.compile(
            r"^(q\d+\.?|question\s+\d+)",
            re.IGNORECASE,
        )

        for line in lines:
            lower = line.lower()

            if (
                "?" in line
                or question_pattern.match(line)
                or lower.startswith(
                    (
                        "what ",
                        "why ",
                        "how ",
                        "define ",
                        "explain ",
                        "describe ",
                        "calculate ",
                        "prove ",
                    )
                )
            ):
                questions.append(line)

        return questions

    def _detect_bullets(
        self,
        lines: List[str],
    ) -> List[str]:
        return [
            line
            for line in lines
            if line.startswith(("-", "•", "*", "–"))
        ]

    def _detect_numbered_items(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        pattern = re.compile(
            r"(?<!\d)(\d{1,2})\s+([A-Z][A-Za-z /&\-]{3,60})\s+[–-]\s+(.+?)(?=\s+\d{1,2}\s+[A-Z][A-Za-z /&\-]{3,60}\s+[–-]|\Z)",
            re.DOTALL,
        )

        return [
            {
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
                "text": " ".join(match.group(3).split()),
            }
            for match in pattern.finditer(text)
        ]

    def _detect_roman_items(
        self,
        text: str,
    ) -> List[str]:
        pattern = re.compile(
            r"\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\.\s+[A-Z][A-Za-z’' /&\-]+"
        )

        return [
            match.group(0).strip()
            for match in pattern.finditer(text)
        ]

    def _detect_key_value_fields(
        self,
        text: str,
    ) -> List[Dict[str, str]]:
        pattern = re.compile(
            r"\b([A-Z][A-Z /-]{3,40})\s*:\s*([^:]{1,120})(?=\s+[A-Z][A-Z /-]{3,40}\s*:|\Z)"
        )

        return [
            {
                "key": match.group(1).strip(),
                "value": " ".join(match.group(2).split()).strip(),
            }
            for match in pattern.finditer(text)
        ]

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

        if "?" in lower or lower.startswith(("what", "why", "how")):
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
