from typing import Dict, Any, List
import re


class ResponsePostprocessor:

    def process(
        self,
        generated_text: str,
        mode: str = "general",
        task: str = "summary",
    ) -> Dict[str, Any]:

        original_text = generated_text or ""
        cleaned_text = original_text

        steps: List[str] = []

        cleaned_text = self._remove_model_wrappers(cleaned_text)
        steps.append("model_wrapper_removal")

        cleaned_text = self._remove_hallucinated_sections(cleaned_text)
        steps.append("hallucinated_section_removal")

        cleaned_text = self._remove_generic_closing_lines(cleaned_text)
        steps.append("generic_closing_removal")

        cleaned_text = self._fix_markdown(cleaned_text)
        steps.append("markdown_fix")

        cleaned_text = self._normalize_bullets(cleaned_text)
        steps.append("bullet_normalization")

        cleaned_text = self._remove_broken_bullets(cleaned_text)
        steps.append("broken_bullet_removal")

        cleaned_text = self._normalize_numbered_lists(cleaned_text)
        steps.append("numbered_list_normalization")

        cleaned_text = self._remove_duplicate_headings(cleaned_text)
        steps.append("duplicate_heading_removal")

        cleaned_text = self._remove_duplicate_lines(cleaned_text)
        steps.append("duplicate_line_removal")

        cleaned_text = self._remove_empty_sections(cleaned_text)
        steps.append("empty_section_removal")

        cleaned_text = self._normalize_spacing(cleaned_text)
        steps.append("spacing_normalization")

        cleaned_text = cleaned_text.strip()

        return {
            "processed_text": cleaned_text,
            "original_length": len(original_text),
            "processed_length": len(cleaned_text),
            "postprocessing_applied": True,
            "postprocessing_steps": steps,
            "mode": mode,
            "task": task,
        }

    def _remove_model_wrappers(
        self,
        text: str,
    ) -> str:

        patterns = [
            r"^\s*Sure[,!.\s]*",
            r"^\s*Here is .*?:\s*",
            r"^\s*Here are .*?:\s*",
            r"^\s*Below is .*?:\s*",
            r"^\s*Of course[,!.\s]*",
            r"^\s*Certainly[,!.\s]*",

            # Gemini / assistant intros
            r"^\s*Hello!?[\s\S]{0,120}?Lumina AI[\s\S]{0,120}?\n+",
            r"^\s*I am Lumina AI[\s\S]{0,120}?\n+",
            r"^\s*As Lumina AI[\s\S]{0,120}?\n+",

            # Generic AI intros
            r"^\s*Here'?s a structured summary[\s\S]{0,80}?\n+",
            r"^\s*Here'?s a concise summary[\s\S]{0,80}?\n+",
            r"^\s*The following notes[\s\S]{0,80}?\n+",
        ]

        cleaned = text

        for pattern in patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        return cleaned

    def _remove_hallucinated_sections(
        self,
        text: str,
    ) -> str:

        hallucinated_patterns = [
            r"#{1,4}\s*Additional Resources[\s\S]*",
            r"#{1,4}\s*Further Reading[\s\S]*",
            r"#{1,4}\s*Recommended Resources[\s\S]*",
            r"#{1,4}\s*External Resources[\s\S]*",
            r"Additional Resources[\s\S]*",
            r"Further Reading[\s\S]*",
            r"Recommended Resources[\s\S]*",
            r"External Resources[\s\S]*",
            r"Online Courses[\s\S]*",
            r"Textbooks[\s\S]*",
            r"Journals[\s\S]*",
            r"Websites[\s\S]*",
        ]

        cleaned = text

        for pattern in hallucinated_patterns:
            cleaned = re.sub(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

        return cleaned

    def _remove_generic_closing_lines(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()
        cleaned_lines = []

        generic_patterns = [
            r"^by following these .*",
            r"^in conclusion[, ].*",
            r"^overall[, ].*",
            r"^this summary .*",
            r"^these notes .*",
            r"^hope this helps.*",
            r"^let me know .*",
        ]

        for line in lines:
            stripped = line.strip()

            should_remove = False

            for pattern in generic_patterns:
                if re.match(
                    pattern,
                    stripped,
                    flags=re.IGNORECASE,
                ):
                    should_remove = True
                    break

            if not should_remove:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _fix_markdown(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"#\s+#",
            "##",
            text,
        )

        text = re.sub(
            r"#+([^\s#])",
            r"# \1",
            text,
        )

        text = re.sub(
            r"^\s*#\s+#\s+",
            "## ",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"\*\*\s*(.*?)\s*\*\*",
            r"**\1**",
            text,
        )

        text = re.sub(
            r"\*{3,}",
            "**",
            text,
        )

        # Convert headings like ## **Title** -> ## Title
        text = re.sub(
            r"^(#{1,6})\s+\*\*(.*?)\*\*$",
            r"\1 \2",
            text,
            flags=re.MULTILINE,
        )

        return text

    def _normalize_bullets(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"^\s*[•●○▪■]\s*",
            "- ",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^\s*[–—]\s+",
            "- ",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^\s*-\s*",
            "- ",
            text,
            flags=re.MULTILINE,
        )

        return text

    def _remove_broken_bullets(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()
        cleaned_lines = []

        broken_values = {
            "-",
            "--",
            "---",
            "- -",
            "- --",
            "- ---",
            "- .",
            "- ,",
            "- ;",
            "- :",
        }

        for line in lines:
            stripped = line.strip()

            if stripped in broken_values:
                continue

            if re.fullmatch(
                r"-\s*[-–—_*.\s]+",
                stripped,
            ):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _normalize_numbered_lists(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"^\s*(\d+)\)\s+",
            r"\1. ",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^\s*(\d+)\.\s*",
            r"\1. ",
            text,
            flags=re.MULTILINE,
        )

        return text

    def _remove_duplicate_headings(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()
        cleaned_lines = []

        seen_headings = set()
        previous_heading = ""

        for line in lines:
            stripped = line.strip()

            heading_text = re.sub(
                r"^[#*\s\d.]+|[#*\s]+$",
                "",
                stripped,
            ).lower()

            is_heading = (
                stripped.startswith("#")
                or (
                    stripped.startswith("**")
                    and stripped.endswith("**")
                )
            )

            if is_heading:
                if heading_text == previous_heading:
                    continue

                if heading_text in seen_headings:
                    continue

                seen_headings.add(heading_text)
                previous_heading = heading_text

            else:
                previous_heading = ""

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _remove_duplicate_lines(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()

        cleaned_lines = []
        seen_recent = set()

        for line in lines:
            normalized = line.strip().lower()

            if not normalized:
                cleaned_lines.append(line)
                continue

            if normalized in seen_recent:
                continue

            cleaned_lines.append(line)
            seen_recent.add(normalized)

            if len(seen_recent) > 120:
                seen_recent.clear()

        return "\n".join(cleaned_lines)

    def _remove_empty_sections(
        self,
        text: str,
    ) -> str:

        lines = text.splitlines()
        cleaned = []

        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            is_heading = stripped.startswith("#")

            if is_heading:
                j = i + 1

                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j >= len(lines):
                    i += 1
                    continue

                if lines[j].strip().startswith("#"):
                    i += 1
                    continue

            cleaned.append(line)
            i += 1

        return "\n".join(cleaned)

    def _normalize_spacing(
        self,
        text: str,
    ) -> str:

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n\s+\n",
            "\n\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()