import re
from typing import Dict, Any, List


class OutputFormatter:

    def format(
        self,
        processed_text: str,
        mode: str = "general",
        task: str = "short_summary",
        model: str = "",
        provider: str = "",
        structure: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        cleaned_markdown = self._clean_markdown(
            processed_text,
        )

        if task == "table_format":
            cleaned_markdown = self._ensure_table_output(
                cleaned_markdown,
                structure or {},
            )

        sections = self._extract_sections(
            cleaned_markdown,
        )

        title = self._extract_title(
            cleaned_markdown,
            sections,
        )

        plain_text = self._markdown_to_plain_text(
            cleaned_markdown,
        )

        return {
            "title": title,
            "mode": mode,
            "task": task,
            "format": self._detect_format(task),
            "markdown": cleaned_markdown,
            "plain_text": plain_text,
            "sections": sections,
            "section_count": len(sections),
            "model": model,
            "provider": provider,
        }

    def _ensure_table_output(
        self,
        text: str,
        structure: Dict[str, Any],
    ) -> str:
        if self._has_valid_markdown_table(text):
            return self._remove_blank_table_rows(text)

        tables = structure.get("tables", [])

        for table in tables:
            rows = table.get("rows", [])
            markdown = self._rows_to_markdown_table(rows)

            if markdown:
                return markdown

        key_values = structure.get("key_value_fields", [])

        if key_values:
            rows = [["Field", "Value"]]

            for item in key_values:
                rows.append(
                    [
                        item.get("key", "Field"),
                        item.get("value", "Not specified"),
                    ]
                )

            markdown = self._rows_to_markdown_table(rows)

            if markdown:
                return markdown

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        rows = [["Item", "Details"]]

        for index, line in enumerate(lines[:20], start=1):
            line = re.sub(r"^[-*\d.]+\s*", "", line).strip()

            if line:
                rows.append([str(index), line])

        markdown = self._rows_to_markdown_table(rows)

        return markdown or text

    def _has_valid_markdown_table(
        self,
        text: str,
    ) -> bool:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        table_lines = [
            line
            for line in lines
            if line.startswith("|") and line.endswith("|")
        ]

        if len(table_lines) < 3:
            return False

        meaningful_rows = [
            line
            for line in table_lines
            if not re.fullmatch(r"\|[\s\-:|]+\|", line)
            and any(cell.strip() for cell in line.strip("|").split("|"))
        ]

        return len(meaningful_rows) >= 2

    def _remove_blank_table_rows(
        self,
        text: str,
    ) -> str:
        cleaned_lines = []

        for line in text.splitlines():
            if line.strip().startswith("|"):
                cells = [
                    cell.strip()
                    for cell in line.strip().strip("|").split("|")
                ]

                if not any(cells):
                    continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def _rows_to_markdown_table(
        self,
        rows: List[List[str]],
    ) -> str:
        if not rows or len(rows) < 2:
            return ""

        width = max(len(row) for row in rows)
        normalized_rows = [
            [str(cell or "Not specified").strip() for cell in row]
            + ["Not specified"] * (width - len(row))
            for row in rows
        ]

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
                    if any(cell and cell != "Not specified" for cell in row)
                ],
            ]
        ).strip()

    def _clean_markdown(
        self,
        text: str,
    ) -> str:

        cleaned = text or ""

        cleaned = re.sub(
            r"#\s+#",
            "#",
            cleaned,
        )

        cleaned = re.sub(
            r"#{3,}\s+",
            "## ",
            cleaned,
        )

        cleaned = re.sub(
            r"^(#{1,6})\s+\*\*(.*?)\*\*$",
            r"\1 \2",
            cleaned,
            flags=re.MULTILINE,
        )

        cleaned = re.sub(
            r"\n{3,}",
            "\n\n",
            cleaned,
        )

        cleaned = re.sub(
            r"[ \t]+",
            " ",
            cleaned,
        )

        return cleaned.strip()

    def _extract_sections(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:

        lines = text.splitlines()

        sections = []
        current_section = None

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            heading = None
            level = None

            markdown_heading = re.match(
                r"^(#{1,4})\s+(.+)$",
                stripped,
            )

            bold_heading = re.match(
                r"^\*\*(.+?)\*\*$",
                stripped,
            )

            if markdown_heading:
                heading = markdown_heading.group(2).strip()
                level = len(markdown_heading.group(1))

            elif bold_heading:
                heading = bold_heading.group(1).strip()
                level = 2

            if heading:
                heading = self._clean_heading_text(
                    heading,
                )

                if not self._is_valid_section_heading(
                    heading,
                ):
                    heading = None
                    level = None

            if heading:
                if current_section:
                    sections.append(current_section)

                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": [],
                }

            else:
                if current_section:
                    current_section["content"].append(stripped)

        if current_section:
            sections.append(current_section)

        return self._remove_empty_or_duplicate_sections(
            sections,
        )

    def _clean_heading_text(
        self,
        heading: str,
    ) -> str:

        heading = re.sub(
            r"^[#\s]+",
            "",
            heading,
        )

        heading = re.sub(
            r"[*_`]",
            "",
            heading,
        )

        return heading.strip()

    def _is_valid_section_heading(
        self,
        heading: str,
    ) -> bool:

        if not heading:
            return False

        if len(heading.split()) > 12:
            return False

        if heading.endswith((".", ",", ";", ":")):
            return False

        if re.match(
            r"^\d+\.\s+[A-Z].*:",
            heading,
        ):
            return False

        return True

    def _remove_empty_or_duplicate_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        cleaned = []
        seen = set()

        for section in sections:
            heading = section.get(
                "heading",
                "",
            ).strip()

            content = section.get(
                "content",
                [],
            )

            normalized_heading = re.sub(
                r"^\d+\.\s*",
                "",
                heading.lower(),
            ).strip()

            has_content = any(
                item.strip()
                for item in content
            )

            if normalized_heading in seen and not has_content:
                continue

            if not has_content and cleaned:
                continue

            seen.add(normalized_heading)
            cleaned.append(section)

        return cleaned

    def _extract_title(
        self,
        text: str,
        sections: List[Dict[str, Any]],
    ) -> str:

        if sections:
            return sections[0]["heading"]

        lines = text.strip().splitlines()

        if not lines:
            return ""

        first_line = lines[0]

        first_line = re.sub(
            r"[*#_`]",
            "",
            first_line,
        ).strip()

        return first_line[:120]

    def _markdown_to_plain_text(
        self,
        text: str,
    ) -> str:

        plain = re.sub(
            r"#{1,6}\s+",
            "",
            text,
        )

        plain = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            plain,
        )

        plain = re.sub(
            r"\*(.*?)\*",
            r"\1",
            plain,
        )

        plain = re.sub(
            r"`(.*?)`",
            r"\1",
            plain,
        )

        plain = re.sub(
            r"\n{3,}",
            "\n\n",
            plain,
        )

        return plain.strip()

    def _detect_format(
        self,
        task: str,
    ) -> str:

        formats = {
            "important_notes": "study_notes",
            "revision_sheet": "revision_sheet",
            "flashcards": "flashcards",
            "mcqs": "mcqs",
            "qa_generation": "question_answers",
            "short_summary": "summary",
            "bullet_summary": "bullet_summary",
            "executive_summary": "executive_summary",
            "meeting_minutes": "meeting_minutes",
        }

        return formats.get(
            task,
            "general_output",
        )
