from typing import Dict, Any


class ModeRouter:
    def route(
        self,
        mode: str,
        task: str,
        pipeline_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        normalized_mode = mode.lower().strip()
        normalized_task = task.lower().strip()

        if normalized_mode == "student":
            return self._student_route(
                normalized_task,
                pipeline_output,
            )

        if normalized_mode == "professional":
            return self._professional_route(
                normalized_task,
                pipeline_output,
            )

        return self._general_route(
            normalized_task,
            pipeline_output,
        )

    def _general_route(
        self,
        task: str,
        pipeline_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        allowed_tasks = [
            "short_summary",
            "bullet_summary",
            "key_points",
            "simplify",
            "clean_text",
        ]

        return {
            "mode": "general",
            "task": task if task in allowed_tasks else "short_summary",
            "model_strategy": "general_summarization",
            "input_text": pipeline_output["final_text"],
            "structure": pipeline_output.get("structure", {}),
            "semantic_chunks": pipeline_output.get("semantic_chunks", {}),
        }

    def _student_route(
        self,
        task: str,
        pipeline_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        allowed_tasks = [
            "important_notes",
            "qa_generation",
            "answer_questions",
            "flashcards",
            "mcqs",
            "beginner_explanation",
            "revision_sheet",
        ]

        return {
            "mode": "student",
            "task": task if task in allowed_tasks else "important_notes",
            "model_strategy": "educational_generation",
            "input_text": pipeline_output["final_text"],
            "structure": pipeline_output.get("structure", {}),
            "semantic_chunks": pipeline_output.get("semantic_chunks", {}),
        }

    def _professional_route(
        self,
        task: str,
        pipeline_output: Dict[str, Any],
    ) -> Dict[str, Any]:

        allowed_tasks = [
            "executive_summary",
            "main_points",
            "action_items",
            "meeting_minutes",
            "structured_report",
            "table_format",
            "email_draft",
        ]

        return {
            "mode": "professional",
            "task": task if task in allowed_tasks else "executive_summary",
            "model_strategy": "professional_document_generation",
            "input_text": pipeline_output["final_text"],
            "structure": pipeline_output.get("structure", {}),
            "semantic_chunks": pipeline_output.get("semantic_chunks", {}),
        }