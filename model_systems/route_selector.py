from typing import Any, Dict


class RouteSelector:
    def select(
        self,
        *,
        features: Dict[str, Any],
        complexity: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = int(complexity.get("score") or 0)
        input_type = features.get("input_type", "plain_text")
        task_type = features.get("task_type", "")
        estimated_ocr_required = bool(features.get("estimated_ocr_required"))
        has_tables = bool(features.get("has_tables"))

        if score >= 62:
            path = "heavy_path"
            model_tier = "heavy"
            parser = "hybrid_or_docling_ready"
            queue_required = True
        elif score >= 28:
            path = "balanced_path"
            model_tier = "medium"
            parser = "rule_or_hybrid"
            queue_required = estimated_ocr_required or has_tables
        else:
            path = "fast_path"
            model_tier = "small"
            parser = "rule"
            queue_required = False

        extractor = self._extractor_for(input_type=input_type, estimated_ocr_required=estimated_ocr_required)

        if task_type in {"short_summary", "bullet_summary", "key_points"} and score < 45:
            model_tier = "small"

        return {
            "path": path,
            "extractor": extractor,
            "parser": parser,
            "model_tier": model_tier,
            "queue_required": queue_required,
            "complexity_score": score,
            "reasons": complexity.get("reasons", []),
        }

    def _extractor_for(self, *, input_type: str, estimated_ocr_required: bool) -> str:
        if input_type == "digital_pdf":
            return "pymupdf"

        if input_type == "scanned_pdf":
            return "pymupdf_or_ocr"

        if input_type == "image" or estimated_ocr_required:
            return "ocr"

        if input_type == "mixed_document":
            return "pymupdf_or_ocr"

        return "direct_text"
