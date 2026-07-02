import os
from typing import Any, Dict, Optional

from model_systems.input.input_analyzer import InputAnalysis


class FeatureExtractor:
    def extract(
        self,
        *,
        text: str = "",
        file_path: Optional[str] = None,
        input_analysis: Optional[InputAnalysis] = None,
        pipeline_output: Optional[Dict[str, Any]] = None,
        mode: str = "general",
        task_type: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        pipeline_output = pipeline_output or {}
        analysis_metadata = input_analysis.metadata if input_analysis else {}
        structure = pipeline_output.get("structure", {})
        extraction = pipeline_output.get("extraction", {})
        semantic_chunks = pipeline_output.get("semantic_chunks", {})
        chunk_metadata = semantic_chunks.get("metadata", {})

        final_text = pipeline_output.get("final_text") or text or ""
        words = final_text.split()
        file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
        page_count = self._first_int(
            analysis_metadata.get("page_count"),
            chunk_metadata.get("page_count"),
            0,
        )

        has_tables = bool(
            analysis_metadata.get("has_tables")
            or structure.get("tables")
            or analysis_metadata.get("table_like_lines", 0) >= 3
        )
        has_images = bool(
            analysis_metadata.get("has_images")
            or analysis_metadata.get("image_count", 0) > 0
        )

        input_type = input_analysis.input_type if input_analysis else "plain_text"
        ocr_confidence = self._first_float(
            extraction.get("confidence"),
            input_analysis.confidence if input_analysis else None,
            0.85,
        )
        layout_complexity = self._layout_complexity(
            page_count=page_count,
            has_tables=has_tables,
            has_images=has_images,
            line_count=self._first_int(analysis_metadata.get("line_count"), 0),
            chunk_count=self._first_int(chunk_metadata.get("chunk_count"), 0),
        )

        return {
            "input_type": input_type,
            "word_count": len(words) or self._first_int(analysis_metadata.get("word_count"), 0),
            "page_count": page_count,
            "file_size": file_size or self._first_int(analysis_metadata.get("file_size"), 0),
            "has_images": has_images,
            "has_tables": has_tables,
            "estimated_ocr_required": self._ocr_required(input_type, analysis_metadata),
            "ocr_confidence": round(ocr_confidence, 3),
            "layout_complexity": layout_complexity,
            "task_type": task_type,
            "mode": mode,
            "user_plan": user_plan,
        }

    def _ocr_required(self, input_type: str, metadata: Dict[str, Any]) -> bool:
        if metadata.get("estimated_ocr_required") is True:
            return True

        return input_type in {"scanned_pdf", "image"}

    def _layout_complexity(
        self,
        *,
        page_count: int,
        has_tables: bool,
        has_images: bool,
        line_count: int,
        chunk_count: int,
    ) -> int:
        score = 0

        if page_count >= 8:
            score += 25
        elif page_count >= 3:
            score += 12

        if has_tables:
            score += 25

        if has_images:
            score += 15

        if line_count >= 100 or chunk_count >= 20:
            score += 20
        elif line_count >= 40 or chunk_count >= 8:
            score += 10

        return min(score, 100)

    def _first_int(self, *values: Any) -> int:
        for value in values:
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                continue

        return 0

    def _first_float(self, *values: Any) -> float:
        for value in values:
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue

        return 0.0
