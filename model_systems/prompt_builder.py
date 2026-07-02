from typing import Any, Dict, List


class PromptBuilder:
    """
    Builds task-aware prompts for short and long documents.

    Important design rules:
    - Every task gets its own instruction, task policy, output shape, and knowledge policy.
    - Tasks that transform source content remain source-grounded.
    - answer_questions may use reliable general knowledge because the source may
      contain questions without answers.
    - Structured outputs adapt to the actual document instead of forcing
      irrelevant fields such as owners, deadlines, or decisions.
    """

    STUDENT_TASKS = {
        "important_notes",
        "qa_generation",
        "answer_questions",
        "flashcards",
        "mcqs",
        "beginner_explanation",
        "revision_sheet",
    }

    PROFESSIONAL_TASKS = {
        "executive_summary",
        "main_points",
        "action_items",
        "meeting_minutes",
        "structured_report",
        "table_format",
        "email_draft",
    }

    GENERAL_TASKS = {
        "short_summary",
        "bullet_summary",
        "key_points",
        "simplify",
        "clean_text",
    }

    def build(
        self,
        mode_output: Dict[str, Any],
    ) -> str:
        mode = str(
            mode_output.get("mode", "general")
            or "general"
        ).strip().lower()

        task = str(
            mode_output.get("task", "short_summary")
            or "short_summary"
        ).strip().lower()

        text = str(
            mode_output.get("input_text", "")
            or mode_output.get("source_text", "")
            or ""
        ).strip()

        structure = mode_output.get(
            "structure",
            {},
        ) or {}

        semantic_chunks = mode_output.get(
            "semantic_chunks",
            {},
        ) or {}

        document_type = self._infer_document_type(
            text=text,
            structure=structure,
        )

        selected_chunks = self._select_relevant_chunks(
            semantic_chunks=semantic_chunks,
            task=task,
            mode=mode,
        )

        structure_summary = self._build_structure_summary(
            structure=structure,
            document_type=document_type,
        )

        chunk_summary = self._build_chunk_summary(
            semantic_chunks=selected_chunks,
        )

        document_context = self._build_document_context(
            semantic_chunks=semantic_chunks,
            document_type=document_type,
        )

        full_content = self._build_safe_full_content(
            text=text,
            semantic_chunks=semantic_chunks,
        )

        if self._should_use_compact_prompt(
            text=text,
            semantic_chunks=semantic_chunks,
            structure=structure,
            task=task,
        ):
            return self._compact_prompt(
                mode=mode,
                task=task,
                text=text,
                document_type=document_type,
            )

        return self._expanded_prompt(
            mode=mode,
            task=task,
            document_type=document_type,
            structure_summary=structure_summary,
            chunk_summary=chunk_summary,
            document_context=document_context,
            full_content=full_content,
        )

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def _compact_prompt(
        self,
        mode: str,
        task: str,
        text: str,
        document_type: str,
    ) -> str:
        return f"""
You are Lumina AI.

Mode:
{mode}

Task:
{task}

Detected document type:
{document_type}

Goal:
{self._mode_goal(mode)}

Task instruction:
{self._task_instruction(mode, task, document_type)}

Task policy:
{self._policy_rules(task)}

Knowledge policy:
{self._knowledge_policy(task)}

Rules:
{self._task_rules(mode, task, document_type)}

Output format:
{self._output_shape(mode, task, document_type)}

Content:
{self._source_text_for_prompt(text)}
""".strip()

    def _expanded_prompt(
        self,
        mode: str,
        task: str,
        document_type: str,
        structure_summary: str,
        chunk_summary: str,
        document_context: str,
        full_content: str,
    ) -> str:
        return f"""
You are Lumina AI.

Mode:
{mode}

Task:
{task}

Detected document type:
{document_type}

Goal:
{self._mode_goal(mode)}

Task instruction:
{self._task_instruction(mode, task, document_type)}

Task policy:
{self._policy_rules(task)}

Knowledge policy:
{self._knowledge_policy(task)}

Rules:
{self._task_rules(mode, task, document_type)}

Output format:
{self._output_shape(mode, task, document_type)}

{document_context}

{structure_summary}

Semantic Chunks:
{chunk_summary}

Full Content:
{full_content}
""".strip()

    def _mode_goal(
        self,
        mode: str,
    ) -> str:
        if mode == "student":
            return (
                "Create an accurate, useful, exam-friendly learning output."
            )

        if mode == "professional":
            return (
                "Create a clear, professional, source-grounded business output."
            )

        return (
            "Create a clean, accurate, and useful response."
        )

    # ------------------------------------------------------------------
    # Task instructions
    # ------------------------------------------------------------------

    def _task_instruction(
        self,
        mode: str,
        task: str,
        document_type: str,
    ) -> str:
        if mode == "student":
            return self._student_task_instruction(
                task=task,
                document_type=document_type,
            )

        if mode == "professional":
            return self._professional_task_instruction(
                task=task,
                document_type=document_type,
            )

        return self._general_task_instruction(
            task=task,
            document_type=document_type,
        )

    def _student_task_instruction(
        self,
        task: str,
        document_type: str,
    ) -> str:
        instructions = {
            "important_notes": (
                "Create complete study notes from the supplied content. "
                "Preserve key concepts, definitions, facts, dates, formulas, "
                "examples, exceptions, and exam-relevant details. Do not omit "
                "short but meaningful lines."
            ),
            "qa_generation": (
                "Generate useful question-answer pairs from facts that are "
                "actually present in the source. Cover the major ideas without "
                "creating questions whose answers require outside knowledge."
            ),
            "answer_questions": (
                "Identify and answer the readable required questions accurately. "
                "Respect explicit source limits such as 'answer any four' or "
                "'attempt any five'; answer exactly that number unless the user "
                "explicitly requests all. The source may contain questions without "
                "answers, so reliable general subject knowledge may be used. "
                "Preserve numbering and section grouping, correct only unambiguous "
                "OCR mistakes, and match answer length strictly to marks or source "
                "instructions. Use 'Question unclear due to OCR' only when the "
                "question itself cannot be understood."
            ),
            "flashcards": (
                "Create compact revision flashcards from the supplied material. "
                "Each card must cover one distinct term, concept, fact, formula, "
                "or relationship. Keep answers concise and accurate."
            ),
            "mcqs": (
                "Create high-quality multiple-choice questions from the supplied "
                "content. Each question must have exactly four options, one correct "
                "answer, and a short explanation. Do not create unsupported facts."
            ),
            "beginner_explanation": (
                "Explain the source in simple language for a beginner while "
                "preserving correct terminology and the original meaning. Explain "
                "difficult terms briefly and use examples only when present or when "
                "a simple generic example is necessary for understanding."
            ),
            "revision_sheet": (
                "Create a compact last-minute revision sheet containing must-know "
                "facts, definitions, formulas, processes, comparisons, exceptions, "
                "and likely exam points that are supported by the source."
            ),
        }

        return instructions.get(
            task,
            "Create clear and accurate study material from the supplied content.",
        )

    def _professional_task_instruction(
        self,
        task: str,
        document_type: str,
    ) -> str:
        instructions = {
            "executive_summary": (
                "Create a concise executive summary covering context, major points, "
                "implications, risks, and next steps only when those elements are "
                "supported by the source."
            ),
            "main_points": (
                "Extract all distinct main points in logical order. Preserve names, "
                "numbers, dates, requirements, decisions, and constraints exactly."
            ),
            "action_items": (
                "Extract only explicit or clearly implied action items. Preserve "
                "the actor or responsible party, timing, contact details, status, "
                "conditions, and notes only when they are relevant and supported "
                "by the source. Do not manufacture actions from descriptive content, "
                "and do not force a fixed project-management schema onto unrelated "
                "document types."
            ),
            "meeting_minutes": (
                "Convert meeting content into minutes containing agenda, attendees, "
                "discussion points, decisions, and action items only when available. "
                "Do not add missing participants, decisions, or deadlines."
            ),
            "structured_report": self._structured_report_instruction(
                document_type=document_type,
            ),
            "table_format": (
                "Convert the supplied content into one or more complete markdown "
                "tables whose rows and columns follow the visible source structure. "
                "Preserve every meaningful field, row, value, unit, reference, and "
                "relationship. Use a key-value table for field-based content, retain "
                "real row-and-column structure for tabular content, and split "
                "unrelated structures into separate tables. Do not invent values, "
                "force irrelevant columns, or fill absent fields with placeholders."
            ),
            "email_draft": (
                "Draft a professional email using only the supplied source details. "
                "Do not invent promises, deadlines, recipients, attachments, or "
                "commitments. Use a neutral placeholder only for missing sender "
                "identity."
            ),
        }

        return instructions.get(
            task,
            "Create a clear professional output grounded in the supplied content.",
        )

    def _structured_report_instruction(
        self,
        document_type: str,
    ) -> str:
        if document_type in {
            "exam_or_questionnaire",
            "questionnaire",
        }:
            return (
                "Organize the source as a structured examination or questionnaire "
                "report. Preserve all instructions, questions, numbering, marks, "
                "topics, dates, institution details, and sections. Do not answer "
                "questions unless answers are present in the source. Do not force "
                "business fields such as owner, deadline, risk, or decision."
            )

        if document_type == "meeting":
            return (
                "Create a structured meeting report using sections appropriate to "
                "the source, such as context, discussion, decisions, action items, "
                "and unresolved points. Include only supported details."
            )

        if document_type == "job_or_hiring":
            return (
                "Create a structured hiring report covering organization, role, "
                "location, eligibility, responsibilities, requirements, benefits, "
                "application details, and deadlines only when present."
            )

        if document_type == "academic_or_notes":
            return (
                "Create a structured academic report with sections that follow the "
                "actual topics and content. Preserve definitions, concepts, methods, "
                "examples, findings, and conclusions only where supported."
            )

        return (
            "Organize the supplied content into a structured report using headings "
            "that match the actual document. Preserve all meaningful facts and do "
            "not force irrelevant sections or missing business fields."
        )

    def _general_task_instruction(
        self,
        task: str,
        document_type: str,
    ) -> str:
        instructions = {
            "short_summary": (
                "Create a concise summary that captures the central meaning and "
                "most important supporting details."
            ),
            "bullet_summary": (
                "Summarize the content as grouped bullet points without losing "
                "important facts, names, dates, numbers, or constraints."
            ),
            "key_points": (
                "Extract the most important distinct points in logical order."
            ),
            "simplify": (
                "Rewrite the content in simpler language without changing its "
                "meaning, facts, uncertainty, or scope."
            ),
            "clean_text": (
                "Clean and organize the text while preserving every meaningful "
                "detail. Fix spacing, obvious OCR joins, headings, lists, and "
                "paragraph boundaries without summarizing or deleting content."
            ),
        }

        return instructions.get(
            task,
            "Create a clear and accurate response from the supplied content.",
        )

    # ------------------------------------------------------------------
    # Knowledge policy and task rules
    # ------------------------------------------------------------------

    def _task_policy(
        self,
        task: str,
    ) -> Dict[str, Any]:
        """
        Return production behavior for one task.

        The policy controls:
        - whether outside knowledge is allowed;
        - whether all meaningful source items must be preserved;
        - how missing fields should be handled;
        - whether tables or diagrams are suitable;
        - the expected level of detail.
        """
        policies: Dict[str, Dict[str, Any]] = {
            # Student tasks
            "important_notes": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
            "qa_generation": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
            "answer_questions": {
                "allow_external_knowledge": True,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": True,
                "verbosity": "source_controlled",
            },
            "flashcards": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "short",
            },
            "mcqs": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
            "beginner_explanation": {
                "allow_external_knowledge": "limited_clarification",
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": True,
                "verbosity": "medium",
            },
            "revision_sheet": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": False,
                "verbosity": "compact",
            },

            # Professional tasks
            "executive_summary": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "short",
            },
            "main_points": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "compact",
            },
            "action_items": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": False,
                "verbosity": "compact",
            },
            "meeting_minutes": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
            "structured_report": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": True,
                "verbosity": "medium",
            },
            "table_format": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": False,
                "verbosity": "compact",
            },
            "email_draft": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },

            # General tasks
            "short_summary": {
                "allow_external_knowledge": False,
                "preserve_all_items": False,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "short",
            },
            "bullet_summary": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "compact",
            },
            "key_points": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "compact",
            },
            "simplify": {
                "allow_external_knowledge": "limited_clarification",
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
            "clean_text": {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": True,
                "allow_diagrams": True,
                "verbosity": "source_preserving",
            },
        }

        return policies.get(
            task,
            {
                "allow_external_knowledge": False,
                "preserve_all_items": True,
                "respect_source_instructions": True,
                "correct_obvious_ocr": True,
                "omit_missing_fields": True,
                "allow_tables": False,
                "allow_diagrams": False,
                "verbosity": "medium",
            },
        )

    def _policy_rules(
        self,
        task: str,
    ) -> str:
        """
        Convert a task policy into explicit model instructions.
        """
        policy = self._task_policy(task)
        rules: List[str] = []

        if policy["respect_source_instructions"]:
            rules.extend(
                [
                    "- Follow explicit source instructions exactly, including item limits, marks, word limits, answer length, and required format.",
                    "- Source-specific instructions override generic mode and task defaults.",
                ]
            )

        if policy["preserve_all_items"]:
            rules.append(
                "- Preserve every meaningful source item unless the source explicitly says to select only some."
            )

        knowledge = policy["allow_external_knowledge"]

        if knowledge is True:
            rules.append(
                "- You may use reliable general knowledge when the task requires answers or explanations not contained in the source."
            )
        elif knowledge == "limited_clarification":
            rules.append(
                "- You may use limited general knowledge only to clarify difficult terms, never to introduce unrelated or unsupported claims."
            )
        else:
            rules.append(
                "- Use only the supplied source content and do not add outside facts."
            )

        if policy["correct_obvious_ocr"]:
            rules.extend(
                [
                    "- Correct obvious OCR spelling mistakes wherever they appear, including headings, questions, labels, and body text.",
                    "- Do not reproduce the corrupted spelling when the intended term is unambiguous.",
                    "- Preserve the original wording when a correction is uncertain.",
                ]
            )

        if policy["omit_missing_fields"]:
            rules.append(
                "- Omit irrelevant or unsupported fields instead of forcing placeholder values."
            )
        else:
            rules.append(
                "- Use 'Not specified' only for required fields or table cells that are relevant but missing."
            )

        if not policy["allow_tables"]:
            rules.append(
                "- Do not use markdown tables unless the task explicitly requires one."
            )

        if policy["allow_diagrams"]:
            rules.extend(
                [
                    "- For diagrams, use an ordinary plain-text layout.",
                    "- Never use triple backticks or fenced code blocks for diagrams.",
                ]
            )

        verbosity_rules = {
            "short": (
                "- Keep the output concise while preserving the essential meaning."
            ),
            "compact": (
                "- Use compact wording and avoid unnecessary explanation."
            ),
            "medium": (
                "- Be complete but avoid unnecessary detail."
            ),
            "source_controlled": (
                "- Match length and depth to source instructions, marks, question type, and requested scope."
            ),
            "source_preserving": (
                "- Preserve the complete meaningful source content; do not summarize or omit details."
            ),
        }

        rules.append(
            verbosity_rules.get(
                policy["verbosity"],
                "- Be concise but complete.",
            )
        )

        return "\n".join(rules)

    def _knowledge_policy(
        self,
        task: str,
    ) -> str:
        if task == "answer_questions":
            return (
                "Use the source to identify the questions. You may use reliable "
                "general subject knowledge to answer them. Do not invent uncertain "
                "facts; clearly flag genuinely ambiguous OCR."
            )

        if task == "beginner_explanation":
            return (
                "Stay grounded in the source. Limited general knowledge may be used "
                "only to clarify a term, never to introduce new unsupported claims."
            )

        return (
            "Use only the supplied source content. Do not add outside facts or "
            "unsupported assumptions."
        )

    def _task_rules(
        self,
        mode: str,
        task: str,
        document_type: str,
    ) -> str:
        rules = [
            "- Start directly with the requested output.",
            "- Do not introduce yourself or describe the processing steps.",
            "- Follow explicit source instructions before generic task defaults.",
            "- Preserve names, dates, numbers, formulas, requirements, ordering, and technical terms.",
            "- Do not omit short but meaningful source details.",
            "- Correct only unambiguous OCR mistakes.",
            "- Keep formatting readable and avoid decorative markdown.",
            "- Do not repeat the same information unnecessarily.",
            "- If chunk limits were applied, state that coverage is limited.",
            "- Ensure the final response is complete and does not end mid-sentence, mid-list, mid-table, or inside an unclosed diagram.",
        ]

        if task == "answer_questions":
            rules.extend(
                [
                    "- Respect explicit exam instructions such as 'answer any four', 'attempt any five', or similar limits.",
                    "- When the source limits the number of optional questions, answer exactly the requested number unless the user explicitly asks for all.",
                    "- Otherwise answer every readable required question.",
                    "- Keep each question separate; never merge unrelated questions.",
                    "- Preserve original question numbering and section grouping when possible.",
                    "- Do not write 'Not clearly available' merely because the source contains no answer.",
                    "- Use 'Question unclear due to OCR' only when the question itself cannot be understood.",
                    "- Match answer length strictly to marks or source instructions.",
                    "- For one-mark or one-to-two-line questions, use one concise sentence where possible and no more than 35 words, except for required lists.",
                    "- For short-note questions, use one compact paragraph or 4-6 concise bullets.",
                    "- Do not add notes explaining that extra optional questions were answered.",
                    "- For diagrams, use a compact plain-text representation without fenced code blocks.",
                ]
            )
        else:
            rules.extend(
                [
                    "- Do not invent facts, examples, names, dates, or numbers.",
                    "- If a required source detail is genuinely absent, omit the irrelevant field instead of forcing 'Not specified'.",
                ]
            )

        if task == "action_items":
            rules.extend(
                [
                    "- Select columns that fit the actual document and extracted actions.",
                    "- Omit unsupported columns instead of filling a wide fixed table with placeholders.",
                    "- Use 'Not specified' only when a selected, relevant field is genuinely required but absent.",
                    "- If no explicit or clearly implied action exists, state that no action item was found instead of inventing one.",
                ]
            )

        if task == "table_format":
            rules.extend(
                [
                    "- Choose columns from the actual source structure rather than a fixed universal schema.",
                    "- For key-value content, prefer a compact Field and Details table.",
                    "- For genuine row-based data, preserve the original row and column relationships.",
                    "- Use separate tables when the source contains unrelated structures.",
                    "- Escape literal pipe characters inside markdown table cells.",
                    "- Do not add rows for fields that are absent from the source.",
                ]
            )

        if task in {"short_summary", "bullet_summary", "key_points"}:
            rules.extend(
                [
                    "- Preserve the central meaning, qualifications, exceptions, and constraints.",
                    "- Do not convert uncertain or conditional statements into definite claims.",
                ]
            )

        if task in {"important_notes", "revision_sheet"}:
            rules.extend(
                [
                    "- Preserve definitions, formulas, exceptions, comparisons, and exam-relevant details when present.",
                    "- Do not create textbook facts that are absent from the supplied material.",
                ]
            )

        if task in {"qa_generation", "flashcards", "mcqs"}:
            rules.extend(
                [
                    "- Generate items only from information supported by the source.",
                    "- Avoid duplicate or trivially reworded items.",
                    "- Cover distinct concepts in proportion to their importance in the source.",
                ]
            )

        if task == "beginner_explanation":
            rules.extend(
                [
                    "- Preserve technical accuracy while simplifying vocabulary and sentence structure.",
                    "- Clearly separate source facts from any limited clarification used for understanding.",
                ]
            )

        if task == "executive_summary":
            rules.extend(
                [
                    "- Prioritize decisions, implications, risks, outcomes, and next steps only when supported.",
                    "- Do not force business-analysis sections that are irrelevant to the document.",
                ]
            )

        if task == "main_points":
            rules.extend(
                [
                    "- Keep each point distinct and non-overlapping.",
                    "- Preserve source order when it carries meaning.",
                ]
            )

        if task == "email_draft":
            rules.extend(
                [
                    "- Infer the email purpose from the source, but do not invent recipients, promises, dates, attachments, or commitments.",
                    "- Use placeholders only when a missing element is structurally necessary.",
                ]
            )

        if task == "simplify":
            rules.extend(
                [
                    "- Preserve every material fact, condition, warning, number, and exception.",
                    "- Simplify language, not meaning or scope.",
                ]
            )

        if task == "meeting_minutes":
            rules.append(
                "- Use 'Not specified' only for a required field that is relevant to the meeting but genuinely missing."
            )

        if task == "structured_report":
            rules.extend(
                [
                    "- Use headings appropriate to the actual source document.",
                    "- Do not force owners, deadlines, decisions, risks, or actions unless relevant and present.",
                    "- If the source is an exam or questionnaire, organize questions instead of answering them.",
                ]
            )

        if task == "clean_text":
            rules.extend(
                [
                    "- Preserve the full content; do not summarize.",
                    "- Correct only obvious formatting and OCR issues.",
                ]
            )

        if mode == "student":
            rules.append(
                "- Use clear, exam-friendly language and preserve academic terminology."
            )

        if mode == "professional":
            rules.append(
                "- Preserve professional metadata only when present and relevant."
            )

        if document_type != "general_document":
            rules.append(
                f"- Adapt headings, fields, and output schema to the detected "
                f"document type: {document_type}."
            )

        return "\n".join(rules)

    # ------------------------------------------------------------------
    # Output shapes
    # ------------------------------------------------------------------

    def _output_shape(
        self,
        mode: str,
        task: str,
        document_type: str,
    ) -> str:
        base_shapes = {
            "important_notes": (
                "Use topic-based headings with concise notes, definitions, key "
                "facts, formulas, examples, comparisons, exceptions, and exam "
                "takeaways only where supported."
            ),
            "qa_generation": (
                "Return a source-proportionate set of distinct question-answer "
                "pairs formatted as Q1./A1., Q2./A2., and so on. Cover major "
                "concepts without duplication."
            ),
            "answer_questions": (
                "Respect each section's source instruction. For short-note "
                "questions, use one compact paragraph or 4-6 concise bullets. "
                "For one-mark or one-to-two-line questions, use no more than two "
                "concise sentences. Format each required item as:\n"
                "Q<number>. <question>\n"
                "Answer: <accurate answer>\n"
                "Keep questions separate and preserve numbering and order."
            ),
            "flashcards": (
                "Return compact cards formatted as:\nTerm: ...\nAnswer: ...\n"
                "Use one distinct concept per card."
            ),
            "mcqs": (
                "Return numbered MCQs with exactly four options A-D, one correct "
                "answer, and a brief source-grounded explanation."
            ),
            "beginner_explanation": (
                "Use clear sections such as Big Idea, Simple Explanation, Key "
                "Terms, Example when genuinely useful, and Quick Recap."
            ),
            "revision_sheet": (
                "Use compact sections such as Must Know, Definitions, Formulas or "
                "Processes, Comparisons, Quick Facts, Common Traps, and Final Recap "
                "only when supported."
            ),
            "executive_summary": (
                "Use only supported sections selected from Context, Key Findings, "
                "Implications, Risks, Decisions, and Next Steps."
            ),
            "main_points": (
                "Return a concise numbered list of distinct, non-overlapping main "
                "points in a logical order."
            ),
            "action_items": self._action_items_shape(
                document_type=document_type,
            ),
            "meeting_minutes": (
                "Use only supported sections selected from Meeting Details, "
                "Attendees, Agenda, Discussion, Decisions, Action Items, and Open "
                "Questions."
            ),
            "structured_report": self._structured_report_shape(
                document_type=document_type,
            ),
            "table_format": self._table_format_shape(
                document_type=document_type,
            ),
            "email_draft": (
                "Return Subject, Greeting, concise Body, Closing, and Signature "
                "Placeholder. Include only details supported by the source."
            ),
            "short_summary": (
                "Return 1-4 concise paragraphs, scaled to source length, preserving "
                "the central meaning and essential supporting details."
            ),
            "bullet_summary": (
                "Return logically grouped bullets with concise headings only when "
                "they improve clarity."
            ),
            "key_points": (
                "Return a source-proportionate numbered list of the most important "
                "distinct points; do not force a fixed count."
            ),
            "simplify": (
                "Rewrite in clear short paragraphs, preserve all material facts and "
                "conditions, and add a brief recap only when useful."
            ),
            "clean_text": (
                "Return the complete cleaned content with preserved headings, "
                "paragraphs, lists, tables, labels, and meaningful details. Do not "
                "summarize."
            ),
        }

        base = base_shapes.get(
            task,
            "Use a clean structure that directly matches the requested task.",
        )

        adaptation = self._document_task_addendum(
            task=task,
            document_type=document_type,
        )

        if not adaptation:
            return base

        return f"{base}\n\nDocument adaptation:\n{adaptation}"

    def _document_task_addendum(
        self,
        task: str,
        document_type: str,
    ) -> str:
        """
        Add document-type guidance without replacing the task's core behavior.
        """
        if document_type == "general_document":
            return ""

        common_guidance = {
            "exam_or_questionnaire": (
                "Preserve section instructions, numbering, marks, limits, and "
                "question order. Do not answer questions unless the selected task "
                "explicitly requires answers."
            ),
            "job_or_hiring": (
                "Preserve organization, roles, location, compensation, eligibility, "
                "requirements, benefits, application method, contact details, and "
                "deadlines only when present."
            ),
            "meeting": (
                "Distinguish discussion, decisions, actions, owners, deadlines, and "
                "open issues; do not merge them."
            ),
            "academic_or_notes": (
                "Preserve topic hierarchy, terminology, definitions, formulas, "
                "examples, methods, findings, and conclusions where present."
            ),
            "policy_or_instructions": (
                "Preserve scope, mandatory requirements, conditions, procedures, "
                "exceptions, responsibilities, warnings, and effective dates."
            ),
            "invoice_or_receipt": (
                "Preserve parties, references, dates, line items, quantities, rates, "
                "taxes, totals, payment terms, and due information exactly."
            ),
            "resume_or_profile": (
                "Preserve profile details, experience, education, skills, projects, "
                "achievements, dates, and contact information without inventing "
                "qualifications."
            ),
            "legal_or_contract": (
                "Preserve parties, obligations, rights, dates, amounts, conditions, "
                "exceptions, termination terms, and jurisdictional wording without "
                "softening legal meaning."
            ),
            "form_or_application": (
                "Preserve field labels, entered values, options, declarations, "
                "required fields, and signature/date areas."
            ),
            "announcement_or_notice": (
                "Preserve the subject, audience, date, location, instructions, "
                "eligibility, deadlines, contact details, and warnings."
            ),
            "technical_document": (
                "Preserve commands, code, parameters, versions, identifiers, "
                "prerequisites, steps, warnings, inputs, and outputs exactly."
            ),
        }

        guidance = common_guidance.get(
            document_type,
            "",
        )

        if task == "email_draft":
            return (
                f"{guidance} Convert only the communication-relevant details into "
                "an email and omit unrelated source structure."
            ).strip()

        if task in {
            "short_summary",
            "bullet_summary",
            "key_points",
            "executive_summary",
            "main_points",
        }:
            return (
                f"{guidance} Prioritize material information while retaining "
                "qualifications and exceptions."
            ).strip()

        return guidance

    def _table_format_shape(
        self,
        document_type: str,
    ) -> str:
        """
        Choose a table structure from document semantics rather than one fixed
        schema.
        """
        if document_type == "invoice_or_receipt":
            return (
                "Return separate markdown tables when useful for document details, "
                "line items, totals, taxes, and payment information. Preserve all "
                "numeric values, units, and references."
            )

        if document_type == "exam_or_questionnaire":
            return (
                "Return a markdown table preserving section, question number, "
                "instruction, question text, marks or limits, and notes where "
                "present. Do not answer questions."
            )

        if document_type == "meeting":
            return (
                "Return separate markdown tables when useful for meeting details, "
                "decisions, action items, and open issues. Use only supported "
                "columns."
            )

        if document_type == "job_or_hiring":
            return (
                "Return one complete markdown table. Use Field and Details columns "
                "for label-rich content, or preserve genuine rows and columns when "
                "the source already has tabular structure. Include all meaningful "
                "role, location, compensation, eligibility, benefit, application, "
                "contact, and deadline details that are present."
            )

        if document_type == "policy_or_instructions":
            return (
                "Return one or more markdown tables organizing requirements, "
                "responsible parties, conditions, procedures, exceptions, triggers, "
                "and deadlines only when present."
            )

        if document_type == "resume_or_profile":
            return (
                "Return separate markdown tables when useful for profile details, "
                "experience, education, skills, projects, and achievements. Preserve "
                "dates and organizations."
            )

        if document_type == "legal_or_contract":
            return (
                "Return one or more markdown tables for parties, obligations, "
                "rights, dates, amounts, conditions, exceptions, and termination "
                "terms. Preserve legal wording and do not infer missing clauses."
            )

        if document_type == "form_or_application":
            return (
                "Return a markdown table preserving each field label, entered value, "
                "selection, requirement status, and notes."
            )

        if document_type == "technical_document":
            return (
                "Return tables that preserve parameters, commands, versions, inputs, "
                "outputs, steps, and constraints. Keep code and identifiers exact."
            )

        return (
            "Return one or more complete markdown tables using columns that match "
            "the visible source structure. Use Field and Details for key-value "
            "content; preserve original rows and columns for true tabular data; use "
            "separate tables for unrelated structures; omit absent fields."
        )

    def _action_items_shape(
        self,
        document_type: str,
    ) -> str:
        """
        Return a document-aware action table without tying the prompt to one
        sample, industry, or workflow.
        """
        if document_type == "meeting":
            return (
                "Use a markdown table with relevant columns selected from "
                "Action, Owner, Deadline, Priority, Status, and Notes. "
                "Keep only columns supported by the source."
            )

        if document_type == "job_or_hiring":
            return (
                "Use a markdown table with relevant columns selected from "
                "Action, Actor, Contact, Deadline, Eligibility or Condition, "
                "and Notes. Do not include unsupported project-management "
                "columns such as priority or status."
            )

        if document_type == "policy_or_instructions":
            return (
                "Use a markdown table with relevant columns selected from "
                "Required Action, Responsible Party, Trigger or Deadline, "
                "Requirement or Condition, and Notes."
            )

        if document_type == "invoice_or_receipt":
            return (
                "Extract only genuine follow-up actions, such as payment, "
                "approval, correction, or submission. Use relevant columns "
                "selected from Action, Responsible Party, Due Date, Amount or "
                "Reference, and Notes. If no action exists, state that no "
                "explicit action item was found."
            )

        if document_type == "exam_or_questionnaire":
            return (
                "Extract only explicit instructions or required tasks. Use "
                "relevant columns selected from Required Task, Section or "
                "Question, Marks or Limit, and Notes. Do not answer questions."
            )

        if document_type == "legal_or_contract":
            return (
                "Extract only explicit obligations, approvals, notices, payments, "
                "renewals, or other required actions. Use relevant columns selected "
                "from Action, Responsible Party, Trigger, Deadline, Clause or "
                "Reference, and Notes."
            )

        if document_type == "form_or_application":
            return (
                "Extract only completion, submission, verification, signature, or "
                "attachment requirements. Use relevant columns selected from Action, "
                "Responsible Party, Required Field or Document, Deadline, and Notes."
            )

        if document_type == "technical_document":
            return (
                "Extract only operational steps, fixes, checks, or implementation "
                "actions. Use relevant columns selected from Action, Component, "
                "Command or Reference, Prerequisite, Expected Result, and Notes."
            )

        return (
            "Use a markdown table containing only relevant columns selected "
            "from Action, Responsible Party or Actor, Deadline or Trigger, "
            "Status, Contact or Reference, and Notes. Omit unsupported columns "
            "rather than filling them with placeholders."
        )

    def _structured_report_shape(
        self,
        document_type: str,
    ) -> str:
        if document_type in {
            "exam_or_questionnaire",
            "questionnaire",
        }:
            return (
                "Use sections such as Document Details, Instructions, Question "
                "Groups, Marks, and Notes. Preserve all questions; do not add answers."
            )

        if document_type == "job_or_hiring":
            return (
                "Use supported sections such as Overview, Role Details, "
                "Requirements, Responsibilities, Benefits, Application Details."
            )

        if document_type == "meeting":
            return (
                "Use supported sections such as Overview, Discussion, Decisions, "
                "Actions, and Open Issues."
            )

        if document_type == "policy_or_instructions":
            return (
                "Use supported sections such as Purpose, Scope, Requirements, "
                "Procedures, Exceptions, Responsibilities, and References."
            )

        if document_type == "invoice_or_receipt":
            return (
                "Use supported sections such as Document Details, Parties, Line "
                "Items, Totals, Payment Information, and Follow-up Requirements."
            )

        if document_type == "resume_or_profile":
            return (
                "Use supported sections such as Profile, Experience, Education, "
                "Skills, Projects, Achievements, and Contact Details."
            )

        if document_type == "legal_or_contract":
            return (
                "Use supported sections such as Parties, Purpose, Definitions, "
                "Obligations, Rights, Financial Terms, Conditions, Exceptions, "
                "Termination, and Governing Terms."
            )

        if document_type == "form_or_application":
            return (
                "Use supported sections such as Applicant or Entity Details, "
                "Required Fields, Selections, Supporting Documents, Declarations, "
                "and Submission Details."
            )

        if document_type == "announcement_or_notice":
            return (
                "Use supported sections such as Subject, Audience, Key Details, "
                "Instructions, Deadlines, Contact Information, and Warnings."
            )

        if document_type == "technical_document":
            return (
                "Use supported sections such as Overview, Prerequisites, Components, "
                "Configuration, Procedure, Inputs and Outputs, Validation, Errors, "
                "and Troubleshooting."
            )

        return (
            "Use source-appropriate headings such as Overview, Main Findings, "
            "Detailed Content, Gaps or Limitations, and Conclusion only when useful."
        )

    # ------------------------------------------------------------------
    # Chunk selection
    # ------------------------------------------------------------------

    def _select_relevant_chunks(
        self,
        semantic_chunks: Dict[str, Any],
        task: str,
        mode: str,
    ) -> Dict[str, Any]:
        chunks = semantic_chunks.get(
            "chunks",
            [],
        ) or []

        metadata = semantic_chunks.get(
            "metadata",
            {},
        ) or {}

        if not chunks:
            return semantic_chunks

        # Never rank away content when all chunks comfortably fit.
        if not metadata.get("limit_applied", False):
            return {
                "chunks": chunks,
                "metadata": metadata,
            }

        preferred = self._preferred_chunk_types(
            mode=mode,
        )

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

    def _preferred_chunk_types(
        self,
        mode: str,
    ) -> List[str]:
        if mode == "student":
            return [
                "full_text",
                "numbered_item",
                "section",
                "key_value",
                "layout_section",
                "paragraph",
            ]

        if mode == "professional":
            return [
                "full_text",
                "table",
                "key_value",
                "section",
                "numbered_item",
                "layout_section",
                "paragraph",
            ]

        return [
            "full_text",
            "section",
            "paragraph",
            "numbered_item",
            "key_value",
            "layout_section",
        ]

    def _resolve_selected_chunk_count(
        self,
        metadata: Dict[str, Any],
        task: str,
        mode: str,
    ) -> int:
        if metadata.get("limit_applied", False):
            return 14

        return 16

    def _chunk_score(
        self,
        chunk: Dict[str, Any],
        task_boost_terms: List[str],
    ) -> int:
        base_score = int(
            chunk.get("priority", 0)
            or 0
        )

        searchable_text = (
            f"{chunk.get('title', '')} "
            f"{chunk.get('content', '')}"
        ).lower()

        boost = sum(
            15
            for term in task_boost_terms
            if term in searchable_text
        )

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
                    "key",
                    "rule",
                    "concept",
                ],
                "answer_questions": [
                    "question",
                    "what",
                    "why",
                    "how",
                    "explain",
                    "define",
                    "name",
                    "draw",
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
                    "question",
                    "instruction",
                ],
                "table_format": [
                    "field",
                    "value",
                    "amount",
                    "date",
                    "total",
                    "role",
                    "location",
                    "requirement",
                ],
                "email_draft": [
                    "contact",
                    "subject",
                    "request",
                    "deadline",
                    "date",
                    "action",
                ],
                "main_points": [
                    "important",
                    "requirement",
                    "decision",
                    "result",
                    "condition",
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

    # ------------------------------------------------------------------
    # Document analysis
    # ------------------------------------------------------------------

    def _infer_document_type(
        self,
        text: str,
        structure: Dict[str, Any],
    ) -> str:
        """
        Infer document type using weighted evidence.

        No single question, heading, keyword, or field is allowed to dominate
        stronger evidence from another document category.
        """
        lower = (text or "").lower()

        metadata = structure.get(
            "metadata",
            {},
        ) or {}

        question_count = int(
            metadata.get("question_count", 0)
            or 0
        )
        table_count = int(
            metadata.get("table_count", 0)
            or 0
        )
        key_value_count = int(
            metadata.get("key_value_count", 0)
            or 0
        )

        scores: Dict[str, int] = {
            "exam_or_questionnaire": 0,
            "job_or_hiring": 0,
            "meeting": 0,
            "academic_or_notes": 0,
            "policy_or_instructions": 0,
            "invoice_or_receipt": 0,
            "resume_or_profile": 0,
            "legal_or_contract": 0,
            "form_or_application": 0,
            "announcement_or_notice": 0,
            "technical_document": 0,
        }

        signal_groups: Dict[str, Dict[str, int]] = {
            "exam_or_questionnaire": {
                "total marks": 5,
                "marks each": 4,
                "answer any": 5,
                "attempt any": 5,
                "write short notes": 5,
                "question paper": 5,
                "answer in one or two lines": 5,
                "q.1": 2,
                "q.2": 2,
            },
            "job_or_hiring": {
                "we are hiring": 5,
                "is hiring": 5,
                "job description": 5,
                "job opening": 4,
                "who can apply": 3,
                "freshers can apply": 4,
                "qualification": 2,
                "responsibilities": 2,
                "salary": 2,
                "stipend": 2,
                "walk-in": 4,
                "walk in": 4,
                "apply now": 3,
                "job referral": 3,
            },
            "meeting": {
                "meeting minutes": 5,
                "minutes of meeting": 5,
                "agenda": 3,
                "attendees": 3,
                "discussion": 2,
                "action item": 3,
                "decision": 2,
            },
            "academic_or_notes": {
                "chapter": 2,
                "definition": 2,
                "theorem": 3,
                "formula": 2,
                "lecture": 3,
                "study notes": 3,
                "learning objective": 3,
            },
            "policy_or_instructions": {
                "policy": 3,
                "terms and conditions": 5,
                "guideline": 3,
                "procedure": 3,
                "must comply": 4,
                "instructions": 2,
                "eligibility criteria": 3,
            },
            "invoice_or_receipt": {
                "invoice": 5,
                "amount due": 5,
                "total amount": 4,
                "bill to": 5,
                "receipt": 5,
                "invoice number": 4,
                "payment due": 4,
            },
            "resume_or_profile": {
                "curriculum vitae": 5,
                "work experience": 4,
                "professional summary": 4,
                "career objective": 3,
                "education": 2,
                "projects": 1,
                "skills": 1,
            },
            "legal_or_contract": {
                "agreement": 3,
                "contract": 5,
                "terms of service": 5,
                "party of the first part": 5,
                "hereinafter": 4,
                "termination": 3,
                "governing law": 5,
                "indemnity": 4,
                "confidentiality clause": 4,
            },
            "form_or_application": {
                "application form": 5,
                "applicant name": 4,
                "required field": 3,
                "declaration": 3,
                "signature": 2,
                "date of birth": 3,
                "attach document": 4,
                "submit application": 4,
            },
            "announcement_or_notice": {
                "important notice": 5,
                "announcement": 4,
                "notice is hereby given": 5,
                "event date": 3,
                "all students are informed": 5,
                "all employees are informed": 5,
                "registration deadline": 3,
            },
            "technical_document": {
                "installation": 2,
                "configuration": 2,
                "api endpoint": 4,
                "error code": 3,
                "prerequisites": 3,
                "command line": 3,
                "environment variable": 4,
                "troubleshooting": 4,
                "expected output": 3,
            },
        }

        for document_type, signals in signal_groups.items():
            for signal, weight in signals.items():
                if signal in lower:
                    scores[document_type] += weight

        if question_count >= 6:
            scores["exam_or_questionnaire"] += 4
        elif question_count >= 3:
            scores["exam_or_questionnaire"] += 2
        elif question_count >= 1:
            scores["exam_or_questionnaire"] += 1

        if table_count > 0 and any(
            term in lower
            for term in [
                "amount",
                "price",
                "total",
                "invoice",
                "receipt",
            ]
        ):
            scores["invoice_or_receipt"] += 2

        if key_value_count >= 3:
            for candidate in (
                "job_or_hiring",
                "invoice_or_receipt",
                "resume_or_profile",
                "policy_or_instructions",
                "legal_or_contract",
                "form_or_application",
                "announcement_or_notice",
                "technical_document",
            ):
                if scores[candidate] > 0:
                    scores[candidate] += 1

        best_type = max(
            scores,
            key=scores.get,
        )
        best_score = scores[best_type]

        if best_score < 3:
            return "general_document"

        sorted_scores = sorted(
            scores.values(),
            reverse=True,
        )
        second_score = (
            sorted_scores[1]
            if len(sorted_scores) > 1
            else 0
        )

        if best_score == second_score and best_score < 5:
            return "general_document"

        return best_type

    # ------------------------------------------------------------------
    # Prompt context helpers
    # ------------------------------------------------------------------

    def _should_use_compact_prompt(
        self,
        text: str,
        semantic_chunks: Dict[str, Any],
        structure: Dict[str, Any],
        task: str,
    ) -> bool:
        metadata = semantic_chunks.get(
            "metadata",
            {},
        ) or {}

        structure_metadata = structure.get(
            "metadata",
            {},
        ) or {}

        word_count = len(text.split())

        if word_count > 1600:
            return False

        if metadata.get("limit_applied") or metadata.get("long_document"):
            return False

        if (
            task == "table_format"
            and structure_metadata.get("table_count", 0) > 0
        ):
            return False

        if (
            word_count > 1200
            and structure_metadata.get("layout_block_count", 0) > 0
        ):
            return False

        return True

    def _build_document_context(
        self,
        semantic_chunks: Dict[str, Any],
        document_type: str,
    ) -> str:
        metadata = semantic_chunks.get(
            "metadata",
            {},
        ) or {}

        return f"""
Document Processing Context:
- Detected document type: {document_type}
- Page count: {metadata.get("page_count", 1)}
- Chunks used: {metadata.get("chunk_count", 0)}
- Original chunks detected: {metadata.get("original_chunk_count", 0)}
- Chunk limit applied: {metadata.get("limit_applied", False)}
- Long document detected: {metadata.get("long_document", False)}
- Recommended strategy: {metadata.get("recommended_strategy", "single_pass_generation")}

Important:
- If a chunk limit was applied, state that the output covers only the processed portion.
- Do not claim full-document coverage when chunks were limited.
""".strip()

    def _build_safe_full_content(
        self,
        text: str,
        semantic_chunks: Dict[str, Any],
    ) -> str:
        metadata = semantic_chunks.get(
            "metadata",
            {},
        ) or {}

        word_count = len(text.split())

        if word_count <= 1800:
            return text

        if metadata.get(
            "limit_applied",
            False,
        ):
            excerpt = " ".join(
                text.split()[:600]
            )
            return (
                "Only a bounded source excerpt is included because chunk limits "
                "were applied. Use the semantic chunks as the primary source and "
                "do not claim full-document coverage.\n\n"
                f"Source excerpt:\n{excerpt}"
            )

        if word_count > 1800:
            return self._source_text_for_prompt(
                text,
            )

        return text

    def _source_text_for_prompt(
        self,
        text: str,
    ) -> str:
        words = text.split()

        if len(words) <= 1800:
            return text

        return " ".join(words[:1800])

    def _build_structure_summary(
        self,
        structure: Dict[str, Any],
        document_type: str,
    ) -> str:
        metadata = structure.get(
            "metadata",
            {},
        ) or {}

        sections = structure.get(
            "sections",
            [],
        ) or []

        questions = structure.get(
            "questions",
            [],
        ) or []

        bullets = structure.get(
            "bullets",
            [],
        ) or []

        paragraphs = structure.get(
            "paragraphs",
            [],
        ) or []

        numbered_items = structure.get(
            "numbered_items",
            [],
        ) or []

        roman_items = structure.get(
            "roman_items",
            [],
        ) or []

        key_value_fields = structure.get(
            "key_value_fields",
            [],
        ) or []

        layout_blocks = structure.get(
            "layout_blocks",
            [],
        ) or []

        section_headings = [
            section.get("heading", "")
            for section in sections
            if section.get("heading")
        ]

        return f"""
Detected Document Structure:
- Document type: {document_type}
- Title: {structure.get("title", "")}
- Sections detected: {metadata.get("section_count", 0)}
- Section headings: {section_headings[:8]}
- Questions detected: {metadata.get("question_count", 0)}
- Bullet points detected: {metadata.get("bullet_count", 0)}
- Numbered items detected: {metadata.get("numbered_item_count", 0)}
- Roman numeral items detected: {metadata.get("roman_item_count", 0)}
- Key-value fields detected: {metadata.get("key_value_count", 0)}
- Paragraphs detected: {metadata.get("paragraph_count", 0)}
- Layout blocks detected: {metadata.get("layout_block_count", 0)}

Extracted samples:
- Questions: {questions[:5]}
- Bullets: {bullets[:5]}
- Numbered items: {numbered_items[:5]}
- Roman items: {roman_items[:5]}
- Key-value fields: {key_value_fields[:6]}
- Paragraphs: {paragraphs[:3]}
- Layout blocks: {self._compact_layout_blocks(layout_blocks)}
""".strip()

    def _build_chunk_summary(
        self,
        semantic_chunks: Dict[str, Any],
    ) -> str:
        chunks = semantic_chunks.get(
            "chunks",
            [],
        ) or []

        metadata = semantic_chunks.get(
            "metadata",
            {},
        ) or {}

        if not chunks:
            return "No semantic chunks detected."

        formatted: List[str] = []

        for index, chunk in enumerate(chunks):
            formatted.append(
                f"""
Chunk {index + 1}
Chunk Type: {chunk.get("chunk_type")}
Title: {chunk.get("title", "")}
Source Index: {chunk.get("source_index", "")}

Content:
{chunk.get("content", "")[:1800]}
""".strip()
            )

        if metadata.get("limit_applied", False):
            formatted.append(
                "Only limited chunks were included. "
                "Do not claim full-document coverage."
            )

        return "\n\n".join(formatted)

    def _compact_layout_blocks(
        self,
        layout_blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        compact_blocks: List[Dict[str, Any]] = []

        for block in layout_blocks[:3]:
            compact_blocks.append(
                {
                    "type": block.get("type"),
                    "bbox": block.get("bbox"),
                    "text": str(
                        block.get("text", "")
                        or ""
                    )[:220],
                }
            )

        return compact_blocks
