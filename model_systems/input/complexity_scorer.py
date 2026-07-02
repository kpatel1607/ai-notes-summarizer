from typing import Any, Dict, List


class ComplexityScorer:
    def score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        score = 0
        reasons: List[str] = []

        word_count = int(features.get("word_count") or 0)
        page_count = int(features.get("page_count") or 0)
        layout_complexity = int(features.get("layout_complexity") or 0)
        ocr_confidence = float(features.get("ocr_confidence") or 0.85)
        input_type = features.get("input_type", "plain_text")
        task_type = features.get("task_type", "")

        if word_count >= 3500 or page_count >= 12:
            score += 25
            reasons.append("very_long_document")
        elif word_count >= 1500 or page_count >= 5:
            score += 20
            reasons.append("long_document")
        elif word_count >= 700 or page_count >= 2:
            score += 10
            reasons.append("medium_document")

        if input_type == "scanned_pdf":
            score += 25
            reasons.append("scanned_pdf")
        elif input_type == "image":
            score += 18
            reasons.append("image_ocr")
        elif input_type == "mixed_document":
            score += 16
            reasons.append("mixed_document")

        if features.get("has_tables"):
            score += 15
            reasons.append("tables_detected")

        if features.get("has_images"):
            score += 10
            reasons.append("images_detected")

        if ocr_confidence < 0.55:
            score += 25
            reasons.append("poor_ocr_confidence")
        elif ocr_confidence < 0.72:
            score += 12
            reasons.append("medium_ocr_confidence")

        if layout_complexity >= 60:
            score += 20
            reasons.append("complex_layout")
        elif layout_complexity >= 30:
            score += 10
            reasons.append("moderate_layout")

        if task_type in {
            "table_format",
            "report",
            "professional_report",
            "proposal",
            "meeting_minutes",
        }:
            score += 10
            reasons.append("structure_sensitive_task")

        if task_type in {"flashcards", "revision_notes", "qa_generation"}:
            score += 6
            reasons.append("study_generation_task")

        return {
            "score": min(score, 100),
            "reasons": reasons,
        }
