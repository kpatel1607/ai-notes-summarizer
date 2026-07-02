import re
from typing import Dict, Any, List, Optional


class OutputFormatter:

    TASK_DEFAULT_TITLES = {
        "important_notes": "Important Notes",
        "revision_sheet": "Revision Sheet",
        "flashcards": "Flashcards",
        "mcqs": "Multiple-Choice Questions",
        "qa_generation": "Questions and Answers",
        "answer_questions": "Answers",
        "beginner_explanation": "Beginner Explanation",
        "executive_summary": "Executive Summary",
        "main_points": "Main Points",
        "meeting_minutes": "Meeting Minutes",
        "action_items": "Action Items",
        "structured_report": "Structured Report",
        "table_format": "Structured Table",
        "email_draft": "Email Draft",
        "short_summary": "Summary",
        "bullet_summary": "Bullet Summary",
        "key_points": "Key Points",
        "simplify": "Simplified Text",
        "clean_text": "Cleaned Text",
    }

    def format(
        self,
        processed_text: str,
        mode: str = "general",
        task: str = "short_summary",
        model: str = "",
        provider: str = "",
        structure: Optional[Dict[str, Any]] = None,
        source_text: str = "",
        route_config: Optional[Dict[str, Any]] = None,
        cached: bool = False,
    ) -> Dict[str, Any]:
        structure = structure or {}
        route_config = route_config or {}

        cleaned_markdown = self._clean_markdown(
            processed_text,
        )

        cleaned_markdown = self._apply_task_shape(
            cleaned_markdown,
            task,
            source_text=source_text,
            structure=structure,
        )

        if task == "table_format":
            cleaned_markdown = self._ensure_table_output(
                cleaned_markdown,
                structure,
            )

        tables = self._extract_markdown_tables(
            cleaned_markdown,
        )

        sections = self._extract_sections(
            cleaned_markdown,
        )

        title = self._resolve_title(
            markdown=cleaned_markdown,
            sections=sections,
            task=task,
            structure=structure,
        )

        plain_text = self._markdown_to_plain_text(
            cleaned_markdown,
        )

        route = str(
            route_config.get(
                "path",
                "",
            )
            or ""
        )
        model_tier = str(
            route_config.get(
                "model_tier",
                "",
            )
            or ""
        )

        return {
            "title": title,
            "mode": mode,
            "task": task,
            "format": self._detect_format(task),
            "markdown": cleaned_markdown,
            "plain_text": plain_text,
            "plainText": plain_text,
            "sections": sections,
            "section_count": len(sections),
            "sectionCount": len(sections),
            "tables": tables,
            "table_count": len(tables),
            "tableCount": len(tables),
            "model": model,
            "provider": provider,
            "route": route,
            "modelTier": model_tier,
            "cached": cached,
            "metadata": {
                "route": route,
                "modelTier": model_tier,
                "cached": cached,
                "sourceTitleUsed": bool(
                    str(
                        structure.get(
                            "title",
                            "",
                        )
                        or ""
                    ).strip()
                ),
                "modelFormattingRequired": (
                    self.requires_model_based_formatting(
                        cleaned_markdown,
                        sections,
                        mode=mode,
                        task=task,
                    )
                ),
            },
        }

    def requires_model_based_formatting(
        self,
        markdown: str,
        sections: List[Dict[str, Any]],
        mode: str = "general",
        task: str = "short_summary",
    ) -> bool:
        cleaned = (markdown or "").strip()

        if not cleaned:
            return True

        if task == "table_format":
            return not self._has_valid_markdown_table(
                cleaned,
            )

        if task == "email_draft":
            return not bool(
                re.search(
                    r"^subject\s*:",
                    cleaned,
                    flags=re.IGNORECASE
                    | re.MULTILINE,
                )
            )

        if task in {
            "action_items",
            "meeting_minutes",
            "structured_report",
            "executive_summary",
            "beginner_explanation",
            "revision_sheet",
        }:
            if sections:
                return False

            if self._has_valid_markdown_table(
                cleaned,
            ):
                return False

            return len(
                cleaned.split()
            ) > 180

        return False

    def _apply_task_shape(
        self,
        text: str,
        task: str,
        source_text: str = "",
        structure: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:
        cleaned = text.strip()
        structure = structure or {}

        if task == "email_draft":
            return self._repair_email_output(
                cleaned,
                source_text,
            )

        if (
            task == "action_items"
            and not self._has_valid_markdown_table(
                cleaned,
            )
        ):
            return self._action_items_to_table(
                cleaned,
                structure=structure,
            )

        if task in {
            "meeting_minutes",
            "structured_report",
            "executive_summary",
            "beginner_explanation",
            "revision_sheet",
        }:
            return self._normalize_existing_headings(
                cleaned,
            )

        return cleaned

    def _normalize_existing_headings(
        self,
        text: str,
    ) -> str:
        lines: List[str] = []

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped:
                lines.append("")
                continue

            if (
                stripped.endswith(":")
                and len(
                    stripped.split()
                )
                <= 10
                and not stripped.lower().startswith(
                    ("http://", "https://")
                )
            ):
                lines.append(
                    f"## {stripped[:-1].strip()}"
                )
            else:
                lines.append(line)

        return "\n".join(
            lines
        ).strip()

    def _repair_email_output(
        self,
        text: str,
        source_text: str = "",
    ) -> str:
        if (
            re.search(r"^subject\s*:", text, flags=re.IGNORECASE | re.MULTILINE)
            and not self._is_empty_email_shell(text)
        ):
            return text

        lines = [
            line.strip(" -")
            for line in text.splitlines()
            if line.strip()
        ]

        body = "\n\n".join(lines) if lines else text.strip()

        if not body or self._is_empty_email_shell(body):
            body = self._email_body_from_source(source_text)

        return "\n\n".join(
            [
                f"Subject: {self._infer_email_subject(source_text)}",
                "Dear Recipient,",
                body,
                "Regards,",
                "[Your Name]",
            ]
        ).strip()

    def _is_empty_email_shell(
        self,
        text: str,
    ) -> bool:
        cleaned_lines = [
            line.strip().lower().strip("[]")
            for line in text.splitlines()
            if line.strip()
        ]

        shell_terms = {
            "subject:",
            "subject: not specified",
            "dear recipient,",
            "dear recipient",
            "hello,",
            "hi,",
            "body",
            "email body",
            "regards,",
            "sincerely,",
            "best regards,",
            "your name",
            "sender name",
        }

        useful_lines = [
            line
            for line in cleaned_lines
            if line not in shell_terms
            and not re.fullmatch(r"subject:\s*", line)
        ]

        useful_words = " ".join(useful_lines).split()

        return len(useful_words) < 8

    def _email_body_from_source(
        self,
        source_text: str,
    ) -> str:
        cleaned = re.sub(r"\s+", " ", source_text or "").strip()

        if not cleaned:
            return "The required email details are not clearly available in the provided content."

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]

        if not sentences:
            sentences = [cleaned]

        body_sentences = sentences[:5]

        return " ".join(body_sentences)

    def _infer_email_subject(
        self,
        source_text: str,
    ) -> str:
        cleaned = re.sub(r"\s+", " ", source_text or "").strip()

        if not cleaned:
            return "Not specified"

        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0]
        subject = first_sentence.strip(" .")

        if len(subject.split()) > 10:
            subject = " ".join(subject.split()[:10])

        return subject[:90] or "Not specified"

    def _action_items_to_table(
        self,
        text: str,
        structure: Optional[
            Dict[str, Any]
        ] = None,
    ) -> str:
        structure = structure or {}

        lines = [
            re.sub(
                r"^[-*\d.)]+\s*",
                "",
                line,
            ).strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return text

        contact_lines = structure.get(
            "contact_lines",
            [],
        ) or []

        default_contact = (
            str(
                contact_lines[0]
            ).strip()
            if contact_lines
            else ""
        )

        rows: List[List[str]] = [
            [
                "Action",
                "Responsible Party or Actor",
                "Deadline or Trigger",
                "Contact or Reference",
                "Notes",
            ]
        ]

        for line in lines[:30]:
            rows.append(
                [
                    line,
                    "",
                    "",
                    default_contact,
                    "",
                ]
            )

        return (
            self._rows_to_markdown_table(
                rows,
                fill_missing=False,
            )
            or text
        )

    def _ensure_named_sections(
        self,
        text: str,
        sections: List[str],
    ) -> str:
        # Compatibility helper. Never invent unavailable sections.
        return self._normalize_existing_headings(
            text,
        )

    def _ensure_table_output(
        self,
        text: str,
        structure: Dict[str, Any],
    ) -> str:
        if self._has_valid_markdown_table(
            text,
        ):
            return self._remove_blank_table_rows(
                text,
            )

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        parsed_rows = self._rows_from_text_patterns(
            lines,
        )

        if parsed_rows:
            markdown = self._rows_to_markdown_table(
                parsed_rows,
                fill_missing=False,
            )

            if markdown:
                return markdown

        for table in structure.get(
            "tables",
            [],
        ) or []:
            rows = table.get(
                "rows",
                [],
            )

            markdown = self._rows_to_markdown_table(
                rows,
                fill_missing=False,
            )

            if markdown:
                return markdown

        key_values = structure.get(
            "key_value_fields",
            [],
        ) or []

        if key_values:
            rows = [
                [
                    "Field",
                    "Details",
                ]
            ]

            for item in key_values:
                key = str(
                    item.get(
                        "key",
                        "",
                    )
                    or ""
                ).strip()
                value = str(
                    item.get(
                        "value",
                        "",
                    )
                    or ""
                ).strip()

                if key and value:
                    rows.append(
                        [
                            key,
                            value,
                        ]
                    )

            markdown = self._rows_to_markdown_table(
                rows,
                fill_missing=False,
            )

            if markdown:
                return markdown

        rows = [
            [
                "Item",
                "Details",
            ]
        ]

        for index, line in enumerate(
            lines[:30],
            start=1,
        ):
            cleaned = re.sub(
                r"^[-*\d.)]+\s*",
                "",
                line,
            ).strip()

            if cleaned:
                rows.append(
                    [
                        str(index),
                        cleaned,
                    ]
                )

        return (
            self._rows_to_markdown_table(
                rows,
                fill_missing=False,
            )
            or text
        )

    def _rows_from_text_patterns(
        self,
        lines: List[str],
    ) -> List[List[str]]:
        key_value_rows = [
            [
                "Field",
                "Details",
            ]
        ]

        for line in lines:
            cleaned = re.sub(
                r"^[-*\d.)]+\s*",
                "",
                line,
            ).strip()

            match = re.match(
                r"^([^:]{1,60})\s*::?\s*(.+)$",
                cleaned,
            )

            if match:
                key = match.group(
                    1
                ).strip()
                value = match.group(
                    2
                ).strip()

                if key and value:
                    key_value_rows.append(
                        [
                            key,
                            value,
                        ]
                    )

        if len(
            key_value_rows
        ) >= 3:
            return key_value_rows

        column_rows: List[List[str]] = []

        for line in lines:
            if (
                line.startswith("|")
                and line.endswith("|")
            ):
                cells = self._split_markdown_table_row(
                    line,
                )

                if cells:
                    column_rows.append(
                        cells
                    )

        if len(
            column_rows
        ) >= 2:
            return column_rows

        return []

    def _has_valid_markdown_table(
        self,
        text: str,
    ) -> bool:
        tables = self._extract_markdown_tables(
            text,
        )

        return bool(
            tables
            and any(
                table.get(
                    "rows",
                )
                for table in tables
            )
        )

    def _remove_blank_table_rows(
        self,
        text: str,
    ) -> str:
        cleaned_lines: List[str] = []

        for line in text.splitlines():
            stripped = line.strip()

            if (
                stripped.startswith("|")
                and stripped.endswith("|")
            ):
                cells = self._split_markdown_table_row(
                    stripped,
                )

                if not any(
                    cell.strip()
                    for cell in cells
                ):
                    continue

            cleaned_lines.append(line)

        return "\n".join(
            cleaned_lines
        ).strip()

    def _rows_to_markdown_table(
        self,
        rows: List[List[str]],
        *,
        fill_missing: bool = False,
    ) -> str:
        if (
            not rows
            or len(rows) < 2
        ):
            return ""

        width = max(
            len(row)
            for row in rows
        )

        normalized_rows: List[
            List[str]
        ] = []

        for row in rows:
            normalized = [
                self._escape_table_cell(
                    str(
                        cell
                        if cell is not None
                        else ""
                    ).strip()
                )
                for cell in row
            ]

            filler = (
                "Not specified"
                if fill_missing
                else ""
            )

            normalized += [
                filler
            ] * (
                width
                - len(normalized)
            )

            normalized_rows.append(
                normalized
            )

        header = normalized_rows[0]
        separator = [
            "---"
        ] * width

        body = [
            row
            for row in normalized_rows[1:]
            if any(
                cell.strip()
                for cell in row
            )
        ]

        if not body:
            return ""

        return "\n".join(
            [
                (
                    "| "
                    + " | ".join(header)
                    + " |"
                ),
                (
                    "| "
                    + " | ".join(separator)
                    + " |"
                ),
                *[
                    (
                        "| "
                        + " | ".join(row)
                        + " |"
                    )
                    for row in body
                ],
            ]
        ).strip()

    def _escape_table_cell(
        self,
        value: str,
    ) -> str:
        value = re.sub(
            r"\r?\n",
            "<br>",
            value,
        )

        value = re.sub(
            r"(?<!\\)\|",
            r"\|",
            value,
        )

        return value.strip()

    def _extract_markdown_tables(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        lines = text.splitlines()
        tables: List[
            Dict[str, Any]
        ] = []
        index = 0

        while index < len(lines):
            line = lines[index].strip()

            if not self._is_markdown_table_line(
                line,
            ):
                index += 1
                continue

            block: List[str] = []

            while index < len(lines):
                candidate = lines[
                    index
                ].strip()

                if not self._is_markdown_table_line(
                    candidate,
                ):
                    break

                block.append(
                    candidate
                )
                index += 1

            if len(block) < 2:
                continue

            parsed_rows = [
                self._split_markdown_table_row(
                    row,
                )
                for row in block
            ]

            parsed_rows = [
                row
                for row in parsed_rows
                if row
            ]

            if not parsed_rows:
                continue

            header = parsed_rows[0]
            body_start = 1

            if (
                len(parsed_rows) > 1
                and self._is_table_separator_row(
                    parsed_rows[1]
                )
            ):
                body_start = 2

            body = [
                row
                for row in parsed_rows[
                    body_start:
                ]
                if not self._is_table_separator_row(
                    row
                )
            ]

            if header and body:
                tables.append(
                    {
                        "headers": header,
                        "rows": body,
                        "row_count": len(
                            body
                        ),
                        "column_count": len(
                            header
                        ),
                    }
                )

        return tables

    def _split_markdown_table_row(
        self,
        line: str,
    ) -> List[str]:
        content = line.strip().strip(
            "|"
        )

        if not content:
            return []

        placeholder = (
            "LUMINATABLEPIPEPLACEHOLDER"
        )

        content = content.replace(
            r"\|",
            placeholder,
        )

        cells = [
            self._clean_table_cell_for_data(
                cell.replace(
                    placeholder,
                    " / ",
                ).strip()
            )
            for cell in content.split(
                "|"
            )
        ]

        return cells

    def _is_table_separator_row(
        self,
        cells: List[str],
    ) -> bool:
        return bool(
            cells
            and all(
                re.fullmatch(
                    r":?-{3,}:?",
                    cell.replace(
                        " ",
                        "",
                    ),
                )
                for cell in cells
            )
        )

    def _clean_table_cell_for_data(
        self,
        value: str,
    ) -> str:
        cleaned = re.sub(
            r"<br\s*/?>",
            "\n",
            value,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"`(.*?)`",
            r"\1",
            cleaned,
        )

        cleaned = cleaned.replace(
            r"\|",
            " / ",
        )

        cleaned = re.sub(
            r"[ \t]{2,}",
            " ",
            cleaned,
        )

        return cleaned.strip()

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
            r"\*\*(.*?)\*\*",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*(?!\*)",
            r"\1",
            cleaned,
        )

        cleaned = re.sub(
            r"^\s*\*\s+",
            "- ",
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
        sections: List[
            Dict[str, Any]
        ] = []
        current_section = None

        for line in lines:
            stripped = line.strip()

            if not stripped:
                continue

            if self._is_markdown_table_line(
                stripped,
            ):
                if current_section:
                    current_section[
                        "content"
                    ].append(stripped)
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
                heading = markdown_heading.group(
                    2
                ).strip()
                level = len(
                    markdown_heading.group(
                        1
                    )
                )

            elif bold_heading:
                heading = bold_heading.group(
                    1
                ).strip()
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
                    sections.append(
                        current_section
                    )

                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": [],
                }

            elif current_section:
                current_section[
                    "content"
                ].append(stripped)

        if current_section:
            sections.append(
                current_section
            )

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
        # Backward-compatible helper.
        return self._resolve_title(
            markdown=text,
            sections=sections,
            task="",
            structure={},
        )

    def _resolve_title(
        self,
        *,
        markdown: str,
        sections: List[Dict[str, Any]],
        task: str,
        structure: Dict[str, Any],
    ) -> str:
        source_title = str(
            structure.get(
                "title",
                "",
            )
            or ""
        ).strip()

        if source_title:
            return self._clean_title(
                source_title,
            )

        for line in markdown.splitlines():
            stripped = line.strip()

            if not stripped:
                continue

            if self._is_markdown_table_line(
                stripped,
            ):
                continue

            heading_match = re.match(
                r"^#{1,6}\s+(.+)$",
                stripped,
            )

            if heading_match:
                return self._clean_title(
                    heading_match.group(
                        1
                    )
                )

            break

        if sections:
            return self._clean_title(
                str(
                    sections[0].get(
                        "heading",
                        "",
                    )
                )
            )

        return self.TASK_DEFAULT_TITLES.get(
            task,
            "Generated Document",
        )

    def _clean_title(
        self,
        title: str,
    ) -> str:
        cleaned = re.sub(
            r"[*#_`]",
            "",
            title or "",
        )

        cleaned = re.sub(
            r"[!?]{1,4}$",
            "",
            cleaned,
        )

        return cleaned.strip()[:120]

    def _is_markdown_table_line(
        self,
        line: str,
    ) -> bool:
        stripped = line.strip()

        return (
            stripped.startswith("|")
            and stripped.endswith("|")
            and stripped.count("|") >= 2
        )

    def _markdown_to_plain_text(
        self,
        text: str,
    ) -> str:
        # Keep <br> inside table cells until after markdown tables are parsed.
        plain = text

        plain = re.sub(
            r"#{1,6}\s+",
            "",
            plain,
        )

        plain = re.sub(
            r"\*\*(.*?)\*\*",
            r"\1",
            plain,
        )

        plain = re.sub(
            r"(?<!\*)\*(.*?)\*(?!\*)",
            r"\1",
            plain,
        )

        plain = re.sub(
            r"`(.*?)`",
            r"\1",
            plain,
        )

        plain = self._markdown_tables_to_plain_text(
            plain,
        )

        plain = re.sub(
            r"<br\s*/?>",
            "\n",
            plain,
            flags=re.IGNORECASE,
        )

        plain = plain.replace(
            r"\|\|",
            " / ",
        )

        plain = plain.replace(
            r"\|",
            " / ",
        )

        plain = re.sub(
            r"[ \t]+\n",
            "\n",
            plain,
        )

        plain = re.sub(
            r"\n{3,}",
            "\n\n",
            plain,
        )

        return plain.strip()

    def _markdown_tables_to_plain_text(
        self,
        text: str,
    ) -> str:
        lines = text.splitlines()
        converted: List[str] = []
        index = 0

        while index < len(lines):
            stripped = lines[
                index
            ].strip()

            if not self._is_markdown_table_line(
                stripped,
            ):
                converted.append(
                    lines[index]
                )
                index += 1
                continue

            block: List[str] = []

            while index < len(lines):
                candidate = lines[
                    index
                ].strip()

                if not self._is_markdown_table_line(
                    candidate,
                ):
                    break

                block.append(
                    candidate
                )
                index += 1

            rows = [
                self._split_markdown_table_row(
                    row
                )
                for row in block
            ]

            rows = [
                row
                for row in rows
                if row
                and not self._is_table_separator_row(
                    row
                )
            ]

            for row in rows:
                converted.append(
                    " | ".join(
                        re.sub(
                            r"\s*\n\s*",
                            "; ",
                            cell,
                        ).strip()
                        for cell in row
                    )
                )

        return "\n".join(
            converted
        )

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
            "answer_questions": "answers",
            "beginner_explanation": "beginner_explanation",
            "short_summary": "summary",
            "bullet_summary": "bullet_summary",
            "key_points": "key_points",
            "executive_summary": "executive_summary",
            "main_points": "main_points",
            "meeting_minutes": "meeting_minutes",
            "action_items": "action_items",
            "structured_report": "structured_report",
            "table_format": "table",
            "email_draft": "email",
            "simplify": "simplified_text",
            "clean_text": "clean_text",
        }

        return formats.get(
            task,
            "general_output",
        )
