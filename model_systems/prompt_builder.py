from typing import Dict, Any, List


class PromptBuilder:
    def build(
        self,
        mode_output: Dict[str, Any],
    ) -> str:

        mode = mode_output["mode"]
        task = mode_output["task"]

        text = mode_output.get(
            "input_text",
            "",
        )

        structure = mode_output.get(
            "structure",
            {},
        )

        semantic_chunks = mode_output.get(
            "semantic_chunks",
            {},
        )

        selected_chunks = self._select_relevant_chunks(
            semantic_chunks=semantic_chunks,
            task=task,
            mode=mode,
        )

        structure_summary = self._build_structure_summary(
            structure=structure,
        )

        chunk_summary = self._build_chunk_summary(
            semantic_chunks=selected_chunks,
        )

        document_context = self._build_document_context(
            semantic_chunks=semantic_chunks,
        )

        full_content = self._build_safe_full_content(
            text=text,
            semantic_chunks=semantic_chunks,
        )

        if mode == "student":
            return self._student_prompt(
                task=task,
                structure_summary=structure_summary,
                chunk_summary=chunk_summary,
                document_context=document_context,
                full_content=full_content,
            )

        if mode == "professional":
            return self._professional_prompt(
                task=task,
                structure_summary=structure_summary,
                chunk_summary=chunk_summary,
                document_context=document_context,
                full_content=full_content,
            )

        return self._general_prompt(
            task=task,
            structure_summary=structure_summary,
            chunk_summary=chunk_summary,
            document_context=document_context,
            full_content=full_content,
        )

    def _select_relevant_chunks(
        self,
        semantic_chunks: Dict[str, Any],
        task: str,
        mode: str,
    ) -> Dict[str, Any]:

        chunks = semantic_chunks.get(
            "chunks",
            [],
        )

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )

        if not chunks:
            return semantic_chunks

        if mode == "student":
            preferred = [
                "numbered_item",
                "section",
                "key_value",
                "layout_section",
                "paragraph",
            ]

        elif mode == "professional":
            preferred = [
                "key_value",
                "section",
                "numbered_item",
                "layout_section",
                "paragraph",
            ]

        else:
            preferred = [
                "section",
                "paragraph",
                "numbered_item",
                "key_value",
                "layout_section",
            ]

        task_boost_terms = self._task_boost_terms(
            task=task,
            mode=mode,
        )

        ranked = sorted(
            chunks,
            key=lambda chunk: (
                chunk.get("chunk_type") not in preferred,
                -self._chunk_score(
                    chunk=chunk,
                    task_boost_terms=task_boost_terms,
                ),
            ),
        )

        max_selected_chunks = self._resolve_selected_chunk_count(
            metadata=metadata,
            task=task,
            mode=mode,
        )

        return {
            "chunks": ranked[:max_selected_chunks],
            "metadata": metadata,
        }

    def _resolve_selected_chunk_count(
        self,
        metadata: Dict[str, Any],
        task: str,
        mode: str,
    ) -> int:

        if metadata.get("limit_applied", False):
            return 8

        if mode == "student" and task in [
            "important_notes",
            "revision_sheet",
            "mcqs",
            "qa_generation",
        ]:
            return 8

        if mode == "professional" and task in [
            "structured_report",
            "meeting_minutes",
            "action_items",
        ]:
            return 8

        return 6

    def _chunk_score(
        self,
        chunk: Dict[str, Any],
        task_boost_terms: List[str],
    ) -> int:

        base_score = int(
            chunk.get(
                "priority",
                0,
            )
        )

        searchable_text = (
            f"{chunk.get('title', '')} "
            f"{chunk.get('content', '')}"
        ).lower()

        boost = 0

        for term in task_boost_terms:
            if term in searchable_text:
                boost += 15

        return base_score + boost

    def _task_boost_terms(
        self,
        task: str,
        mode: str,
    ) -> List[str]:

        if mode == "student":
            return {
                "important_notes": [
                    "definition",
                    "objective",
                    "important",
                    "chapter",
                    "abstract",
                    "key",
                    "rule",
                    "concept",
                ],
                "revision_sheet": [
                    "definition",
                    "formula",
                    "key",
                    "important",
                    "objective",
                ],
                "mcqs": [
                    "definition",
                    "concept",
                    "objective",
                    "important",
                ],
                "qa_generation": [
                    "question",
                    "definition",
                    "explain",
                    "objective",
                ],
                "flashcards": [
                    "definition",
                    "term",
                    "concept",
                    "key",
                ],
            }.get(
                task,
                [
                    "definition",
                    "important",
                    "objective",
                    "key",
                ],
            )

        if mode == "professional":
            return {
                "action_items": [
                    "deadline",
                    "owner",
                    "action",
                    "must",
                    "required",
                    "priority",
                ],
                "meeting_minutes": [
                    "agenda",
                    "decision",
                    "action",
                    "deadline",
                    "discussion",
                ],
                "executive_summary": [
                    "objective",
                    "result",
                    "risk",
                    "decision",
                    "summary",
                ],
                "structured_report": [
                    "requirement",
                    "objective",
                    "result",
                    "method",
                    "conclusion",
                ],
            }.get(
                task,
                [
                    "objective",
                    "requirement",
                    "decision",
                    "action",
                ],
            )

        return [
            "important",
            "summary",
            "key",
            "main",
        ]

    def _build_document_context(
        self,
        semantic_chunks: Dict[str, Any],
    ) -> str:

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )

        return f"""
Document Processing Context:
- Page count: {metadata.get("page_count", 1)}
- Chunks used: {metadata.get("chunk_count", 0)}
- Original chunks detected: {metadata.get("original_chunk_count", 0)}
- Free-tier limit applied: {metadata.get("limit_applied", False)}
- Long document detected: {metadata.get("long_document", False)}
- Recommended strategy: {metadata.get("recommended_strategy", "single_pass_generation")}

Important:
- If free-tier limit is applied, clearly mention that the output is based only on the processed portion.
- Do not claim full-document coverage if chunks were limited.
"""

    def _build_safe_full_content(
        self,
        text: str,
        semantic_chunks: Dict[str, Any],
    ) -> str:

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )

        limit_applied = metadata.get(
            "limit_applied",
            False,
        )

        word_count = len(
            text.split()
        )

        if limit_applied or word_count > 700:
            return (
                "Full content omitted to reduce token usage. "
                "Use only the semantic chunks and structure summary above."
            )

        return text[:1500]

    def _build_structure_summary(
        self,
        structure: Dict[str, Any],
    ) -> str:

        metadata = structure.get(
            "metadata",
            {},
        )

        sections = structure.get(
            "sections",
            [],
        )

        questions = structure.get(
            "questions",
            [],
        )

        bullets = structure.get(
            "bullets",
            [],
        )

        paragraphs = structure.get(
            "paragraphs",
            [],
        )

        numbered_items = structure.get(
            "numbered_items",
            [],
        )

        roman_items = structure.get(
            "roman_items",
            [],
        )

        key_value_fields = structure.get(
            "key_value_fields",
            [],
        )

        layout_blocks = structure.get(
            "layout_blocks",
            [],
        )

        section_headings = [
            section.get(
                "heading",
                "",
            )
            for section in sections
            if section.get("heading")
        ]

        layout_types = metadata.get(
            "layout_types",
            [],
        )

        return f"""
Detected Document Structure:
- Title: {structure.get("title", "")}
- Sections detected: {metadata.get("section_count", 0)}
- Section headings: {section_headings[:6]}
- Questions detected: {metadata.get("question_count", 0)}
- Bullet points detected: {metadata.get("bullet_count", 0)}
- Numbered items detected: {metadata.get("numbered_item_count", 0)}
- Roman numeral items detected: {metadata.get("roman_item_count", 0)}
- Key-value fields detected: {metadata.get("key_value_count", 0)}
- Paragraphs detected: {metadata.get("paragraph_count", 0)}
- Layout blocks detected: {metadata.get("layout_block_count", 0)}
- Layout types: {layout_types}
- Repeated noise removed: {metadata.get("noise_removed", False)}
- Removed noise count: {metadata.get("removed_noise_count", 0)}

Important Extracted Elements:
- Questions sample: {questions[:3]}
- Bullet sample: {bullets[:3]}
- Numbered item sample: {numbered_items[:4]}
- Roman item sample: {roman_items[:5]}
- Key-value sample: {key_value_fields[:5]}
- Paragraph sample: {paragraphs[:2]}
- Layout block sample: {self._compact_layout_blocks(layout_blocks)}
"""

    def _build_chunk_summary(
        self,
        semantic_chunks: Dict[str, Any],
    ) -> str:

        chunks = semantic_chunks.get(
            "chunks",
            [],
        )

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )

        if not chunks:
            return "No semantic chunks detected."

        formatted = []

        for index, chunk in enumerate(chunks):
            formatted.append(
                f"""
Chunk {index + 1}
Chunk Type: {chunk.get("chunk_type")}
Title: {chunk.get("title", "")}
Source Index: {chunk.get("source_index", "")}

Content:
{chunk.get("content", "")[:600]}
"""
            )

        if metadata.get(
            "limit_applied",
            False,
        ):
            formatted.append(
                """
NOTE:
Only limited chunks were included due to the current user plan or token budget.
Do not claim full-document coverage.
"""
            )

        return "\n".join(formatted)

    def _compact_layout_blocks(
        self,
        layout_blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        compact_blocks = []

        for block in layout_blocks[:2]:
            compact_blocks.append(
                {
                    "type": block.get("type"),
                    "bbox": block.get("bbox"),
                    "text": block.get("text", "")[:180],
                }
            )

        return compact_blocks

    def _student_prompt(
        self,
        task: str,
        structure_summary: str,
        chunk_summary: str,
        document_context: str,
        full_content: str,
    ) -> str:

        task_instruction = self._student_task_instruction(
            task,
        )

        return f"""
You are Lumina AI in Student Mode.

Task:
{task}

Goal:
Create a useful learning output for a student.

Task-specific instruction:
{task_instruction}

Rules:
- Start directly with the output. Do not introduce yourself.
- Do not say "Here is", "Hello", or "I have processed".
- Use simple, exam-friendly language.
- Preserve important concepts, definitions, rules, numbers, and requirements from the provided chunks.
- Explain difficult terms briefly only if needed.
- Use clear headings and bullet points.
- If the content is guidelines/instructions, convert them into checklist-style notes.
- Avoid repeating the same information across sections.
- Merge overlapping points into one concise explanation.
- Ignore repeated page headers, footers, watermarks, page numbers, and copyright-like noise unless meaningful.
- If information is missing or unclear, say it is not clearly available in the processed content.
- Do not add textbook names, external resources, examples, facts, or assumptions not present in the content.
- Do not use markdown tables unless explicitly requested.
- If chunk limit was applied, clearly mention: "This output is based on the processed portion of the document."

{document_context}

{structure_summary}

Semantic Chunks:
{chunk_summary}

Full Content:
{full_content}
"""

    def _professional_prompt(
        self,
        task: str,
        structure_summary: str,
        chunk_summary: str,
        document_context: str,
        full_content: str,
    ) -> str:

        task_instruction = self._professional_task_instruction(
            task,
        )

        return f"""
You are Lumina AI in Professional Mode.

Task:
{task}

Goal:
Create a professional, concise, business-ready output.

Task-specific instruction:
{task_instruction}

Rules:
- Start directly with the output. Do not introduce yourself.
- Do not say "Here is", "Hello", or "I have processed".
- Keep the output clear and structured.
- Preserve useful professional metadata such as names, dates, roles, company names, contact details, requirements, deadlines, and legal/compliance points if present.
- Extract decisions, risks, priorities, requirements, deadlines, and action items when relevant.
- If the document is a resume, preserve candidate identity, contact information, skills, education, projects, achievements, and experience.
- If the content is policy/guidelines, convert it into requirements and checklist sections.
- Avoid repeating the same information across sections.
- Merge overlapping points into one concise explanation.
- Ignore repeated decorative page headers/footers unless they contain useful professional metadata.
- If information is missing or unclear, say it is not clearly available in the processed content.
- Do not add unsupported information.
- Do not exaggerate or assume missing details.
- Do not use markdown tables unless explicitly requested.
- If chunk limit was applied, clearly mention that the output is based only on the processed portion.

{document_context}

{structure_summary}

Semantic Chunks:
{chunk_summary}

Full Content:
{full_content}
"""

    def _general_prompt(
        self,
        task: str,
        structure_summary: str,
        chunk_summary: str,
        document_context: str,
        full_content: str,
    ) -> str:

        task_instruction = self._general_task_instruction(
            task,
        )

        return f"""
You are Lumina AI in General Mode.

Task:
{task}

Goal:
Create a clean and useful response.

Task-specific instruction:
{task_instruction}

Rules:
- Start directly with the output. Do not introduce yourself.
- Do not say "Here is", "Hello", or "I have processed".
- Be concise but complete.
- Keep the original meaning.
- Use readable formatting.
- Preserve important details.
- Avoid repeating the same information across sections.
- Merge overlapping points into one concise explanation.
- Ignore repeated page headers, footers, page numbers, and watermarks unless relevant.
- If information is missing or unclear, say it is not clearly available in the processed content.
- Do not invent details.
- Do not use markdown tables unless explicitly requested.
- If chunk limit was applied, clearly mention that the output is based only on the processed portion.

{document_context}

{structure_summary}

Semantic Chunks:
{chunk_summary}

Full Content:
{full_content}
"""

    def _student_task_instruction(
        self,
        task: str,
    ) -> str:

        instructions = {
            "important_notes": "Create important notes with headings, definitions, key points, and exam-focused takeaways.",
            "qa_generation": "Generate useful question-answer pairs from the content.",
            "answer_questions": "Answer only the questions found in the content using the provided material.",
            "flashcards": "Create flashcards in Q/A format for revision.",
            "mcqs": "Create multiple-choice questions with four options and mark the correct answer.",
            "beginner_explanation": "Explain the content as if teaching a beginner.",
            "revision_sheet": "Create a compact revision sheet with definitions, key facts, formulas if present, and quick review points.",
        }

        return instructions.get(
            task,
            "Create clear study notes from the content.",
        )

    def _professional_task_instruction(
        self,
        task: str,
    ) -> str:

        instructions = {
            "executive_summary": "Create an executive summary with context, key points, and conclusion.",
            "main_points": "Extract the main points in a clean numbered list.",
            "action_items": "Extract action items, owners if mentioned, deadlines if mentioned, and priority.",
            "meeting_minutes": "Create meeting minutes with agenda, discussion points, decisions, and action items.",
            "structured_report": "Convert the content into a structured report with headings and concise sections.",
            "table_format": "Convert suitable information into clean tables.",
            "email_draft": "Draft a professional email based only on the provided content.",
        }

        return instructions.get(
            task,
            "Create a professional structured summary.",
        )

    def _general_task_instruction(
        self,
        task: str,
    ) -> str:

        instructions = {
            "short_summary": "Create a short summary of the content.",
            "bullet_summary": "Summarize the content as bullet points.",
            "key_points": "Extract the most important key points.",
            "simplify": "Rewrite the content in simpler language.",
            "clean_text": "Clean and organize the content without changing its meaning.",
        }

        return instructions.get(
            task,
            "Summarize the content clearly.",
        )