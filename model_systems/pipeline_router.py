from typing import Dict, Any

from model_systems.input_understanding import (
    InputUnderstandingSystem,
)

from model_systems.ocr_pipeline import OCRPipeline

from model_systems.text_cleanup_pipeline import (
    TextCleanupPipeline,
)

from model_systems.document_structure_parser import (
    DocumentStructureParser,
)

from model_systems.mode_router import ModeRouter

from model_systems.prompt_builder import PromptBuilder

from model_systems.semantic_chunker import SemanticChunker

from model_systems.generation_service import GenerationService

from model_systems.smart_text_normalizer import SmartTextNormalizer

from model_systems.response_postprocessor import (
    ResponsePostprocessor,
)

from model_systems.output_formatter import (
    OutputFormatter,
)

# Future use:
# from model_systems.ocr_correction_model import (
#     OCRCorrectionModel,
# )


class PipelineRouter:
    def __init__(self):
        self.input_system = InputUnderstandingSystem()
        self.ocr_pipeline = OCRPipeline()
        self.cleanup_pipeline = TextCleanupPipeline()
        self.structure_parser = DocumentStructureParser()
        self.mode_router = ModeRouter()
        self.prompt_builder = PromptBuilder()
        self.semantic_chunker = SemanticChunker()
        self.generation_service = GenerationService()
        self.smart_normalizer = SmartTextNormalizer()
        self.response_postprocessor = (
            ResponsePostprocessor()
        )
        self.output_formatter = OutputFormatter()

        # Future use:
        # self.ocr_correction_model = OCRCorrectionModel()

    def process_text(
        self,
        text: str,
    ) -> Dict[str, Any]:

        understanding = self.input_system.analyze_text(text)

        extracted = self.ocr_pipeline.extract_from_text(text)

        cleaned = self.cleanup_pipeline.clean_ocr_text(
            extracted["text"],
        )

        normalized = self.smart_normalizer.normalize(
            cleaned["cleaned_text"],
        )

        structure = self.structure_parser.parse(
            normalized["normalized_text"],
        )
        semantic_chunks = self.semantic_chunker.chunk(
            {
                "final_text": normalized["normalized_text"],
                "normalization": normalized,
                "structure": structure,
                "understanding": understanding.__dict__,
            }
        )

        return {
            "understanding": understanding.__dict__,
            "semantic_chunks": semantic_chunks,
            "extraction": extracted,
            "cleanup": cleaned,
            "normalization": normalized,
            "structure": structure,
            "final_text": normalized["normalized_text"],
        }

    def process_file(
            self,
            file_path: str,
    ) -> Dict[str, Any]:

        understanding = self.input_system.analyze_file(
            file_path,
        )

        strategy = understanding.extraction_strategy

        layout_structure = None

        if strategy == "pdf_text_extraction":
            extracted = self.ocr_pipeline.extract_from_pdf(
                file_path,
            )

        elif strategy in [
            "pdf_to_image_ocr",
            "ocr_fallback",
        ]:
            extracted = self.ocr_pipeline.extract_from_scanned_pdf(
                file_path,
            )

        elif strategy in [
            "image_ocr",
            "enhanced_image_ocr",
            "image_preprocessing_then_ocr",
        ]:
            extracted = self.ocr_pipeline.extract_from_image(
                file_path,
            )

            layout_structure = self.structure_parser.parse_document_image(
                file_path,
            )

        else:
            extracted = {
                "text": "",
                "source": "unsupported",
                "confidence": 0.0,
            }

        cleaned = self.cleanup_pipeline.clean_ocr_text(
            extracted["text"],
        )

        normalized = self.smart_normalizer.normalize(
            cleaned["cleaned_text"],
        )

        text_structure = self.structure_parser.parse(
            normalized["normalized_text"],
        )

        structure = self.structure_parser.merge_text_and_layout_structure(
            text_structure=text_structure,
            layout_structure=layout_structure,
        )

        semantic_chunks = self.semantic_chunker.chunk(
            {
                "final_text": normalized["normalized_text"],
                "normalization": normalized,
                "structure": structure,
                "understanding": understanding.__dict__,
            }
        )

        return {
            "understanding": understanding.__dict__,
            "semantic_chunks": semantic_chunks,
            "extraction": extracted,
            "cleanup": cleaned,
            "normalization": normalized,
            "structure": structure,
            "final_text": normalized["normalized_text"],
        }

    def test_layout_parser(
        self,
        image_path: str,
    ) -> Dict[str, Any]:

        layout_structure = self.structure_parser.parse_document_image(
            image_path,
        )

        return {
            "file_path": image_path,
            "layout_structure": layout_structure,
        }

    def process_text_for_mode(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
    ) -> Dict[str, Any]:

        pipeline_output = self.process_text(text)

        mode_output = self.mode_router.route(
            mode=mode,
            task=task,
            pipeline_output=pipeline_output,
        )

        generation_prompt = self.prompt_builder.build(
            mode_output,
        )

        return {
            "pipeline_output": pipeline_output,
            "mode_output": mode_output,
            "generation_prompt": generation_prompt,
        }

    def process_file_for_mode(
        self,
        file_path: str,
        mode: str = "general",
        task: str = "short_summary",
    ) -> Dict[str, Any]:

        pipeline_output = self.process_file(file_path)

        mode_output = self.mode_router.route(
            mode=mode,
            task=task,
            pipeline_output=pipeline_output,
        )

        generation_prompt = self.prompt_builder.build(
            mode_output,
        )

        return {
            "pipeline_output": pipeline_output,
            "mode_output": mode_output,
            "generation_prompt": generation_prompt,
        }

    def _generate_with_strategy(
            self,
            prepared: Dict[str, Any],
            mode: str,
            task: str,
    ) -> Dict[str, Any]:

        pipeline_output = prepared.get(
            "pipeline_output",
            {},
        )

        semantic_chunks = pipeline_output.get(
            "semantic_chunks",
            {},
        )

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )

        strategy = metadata.get(
            "recommended_strategy",
            "single_pass_generation",
        )

        if strategy == "hierarchical_summary":
            return self._generate_hierarchical_summary(
                prepared=prepared,
                mode=mode,
                task=task,
            )

        return self._generate_single_pass(
            prepared=prepared,
            mode=mode,
            task=task,
        )

    def _generate_single_pass(
            self,
            prepared: Dict[str, Any],
            mode: str,
            task: str,
    ) -> Dict[str, Any]:

        generation_result = self.generation_service.generate(
            prepared["generation_prompt"],
            max_tokens=2500,
        )

        postprocessed = self.response_postprocessor.process(
            generated_text=generation_result.get(
                "generated_text",
                "",
            ),
            mode=mode,
            task=task,
        )

        formatted_output = self.output_formatter.format(
            processed_text=postprocessed[
                "processed_text"
            ],
            mode=mode,
            task=task,
            model=generation_result.get(
                "model",
                "",
            ),
            provider=generation_result.get(
                "provider",
                "",
            ),
        )

        return {
            "generation_strategy": "single_pass_generation",
            "generation_result": generation_result,
            "postprocessed_output": postprocessed,
            "formatted_output": formatted_output,
        }

    def _generate_hierarchical_summary(
            self,
            prepared: Dict[str, Any],
            mode: str,
            task: str,
    ) -> Dict[str, Any]:

        pipeline_output = prepared.get(
            "pipeline_output",
            {},
        )

        semantic_chunks = pipeline_output.get(
            "semantic_chunks",
            {},
        )

        chunks = semantic_chunks.get(
            "chunks",
            [],
        )

        metadata = semantic_chunks.get(
            "metadata",
            {},
        )


        chunk_summaries = []

        for index, chunk in enumerate(chunks):
            chunk_prompt = self._build_chunk_summary_prompt(
                chunk=chunk,
                index=index,
                mode=mode,
                task=task,
            )

            chunk_generation = self.generation_service.generate(
                chunk_prompt,
                max_tokens=350,
            )

            if not chunk_generation.get("success"):
                continue

            chunk_summaries.append(
                {
                    "chunk_index": index,
                    "chunk_type": chunk.get(
                        "chunk_type",
                        "",
                    ),
                    "title": chunk.get(
                        "title",
                        "",
                    ),
                    "success": chunk_generation.get(
                        "success",
                        False,
                    ),
                    "summary": chunk_generation.get(
                        "generated_text",
                        "",
                    ),
                    "provider": chunk_generation.get(
                        "provider",
                        "",
                    ),
                    "model": chunk_generation.get(
                        "model",
                        "",
                    ),
                }
            )

        merge_prompt = self._build_merge_summary_prompt(
            chunk_summaries=chunk_summaries,
            metadata=metadata,
            mode=mode,
            task=task,
        )

        final_generation = self.generation_service.generate(
            merge_prompt,
            max_tokens=1400,
        )

        postprocessed = self.response_postprocessor.process(
            generated_text=final_generation.get(
                "generated_text",
                "",
            ),
            mode=mode,
            task=task,
        )

        formatted_output = self.output_formatter.format(
            processed_text=postprocessed[
                "processed_text"
            ],
            mode=mode,
            task=task,
            model=final_generation.get(
                "model",
                "",
            ),
            provider=final_generation.get(
                "provider",
                "",
            ),
        )

        return {
            "generation_strategy": "hierarchical_summary",
            "chunk_summaries": chunk_summaries,
            "generation_result": final_generation,
            "postprocessed_output": postprocessed,
            "formatted_output": formatted_output,
            "free_tier_limit_response": self._build_limit_response(
                metadata,
            ),
        }

    def _build_chunk_summary_prompt(
            self,
            chunk: Dict[str, Any],
            index: int,
            mode: str,
            task: str,
    ) -> str:

        return f"""
You are Lumina AI.

Summarize this document chunk for later merging.

Mode:
{mode}

Final user task:
{task}

Chunk number:
{index + 1}

Chunk type:
{chunk.get("chunk_type", "")}

Chunk title:
{chunk.get("title", "")}

Rules:
- Summarize only the provided chunk.
- Preserve important facts, definitions, requirements, dates, numbers, and instructions.
- Do not add outside information.
- Keep the summary concise but complete.

Chunk content:
{chunk.get("content", "")}
"""

    def _build_merge_summary_prompt(
            self,
            chunk_summaries,
            metadata: Dict[str, Any],
            mode: str,
            task: str,
    ) -> str:

        summaries_text = "\n\n".join(
            [
                f"""
Chunk {item["chunk_index"] + 1}
Title: {item["title"]}
Summary:
{item["summary"]}
"""
                for item in chunk_summaries
                if item.get("summary")
            ]
        )

        limit_note = ""

        if metadata.get("limit_applied"):
            limit_note = """
Important limitation:
Only part of the document was processed due to the current free-tier limit.
You must clearly mention that the output is based on the processed portion only.
"""

        return f"""
You are Lumina AI.

Create the final output from these chunk summaries.

Mode:
{mode}

Task:
{task}

Document metadata:
- Page count: {metadata.get("page_count")}
- Chunks used: {metadata.get("chunk_count")}
- Original chunks detected: {metadata.get("original_chunk_count")}
- Limit applied: {metadata.get("limit_applied")}
- User plan: {metadata.get("user_plan")}

{limit_note}

Rules:
- Use only the chunk summaries below.
- Do not add unsupported facts.
- Avoid repetition.
- Create a clean, structured final answer.
- Match the requested mode and task.

Chunk summaries:
{summaries_text}
"""

    def _build_limit_response(
            self,
            metadata: Dict[str, Any],
    ) -> Dict[str, Any]:

        limit_applied = metadata.get(
            "limit_applied",
            False,
        )

        if not limit_applied:
            return {
                "limit_applied": False,
                "message": "",
            }

        return {
            "limit_applied": True,
            "message": (
                "This output is based on the processed portion of the document "
                "because the current free-tier chunk limit was reached."
            ),
            "chunks_used": metadata.get(
                "chunk_count",
                0,
            ),
            "original_chunks": metadata.get(
                "original_chunk_count",
                0,
            ),
            "user_plan": metadata.get(
                "user_plan",
                "free",
            ),
        }

    def generate_from_file(
            self,
            file_path: str,
            mode: str = "general",
            task: str = "short_summary",
    ) -> Dict[str, Any]:

        prepared = self.process_file_for_mode(
            file_path=file_path,
            mode=mode,
            task=task,
        )

        generation_payload = self._generate_with_strategy(
            prepared=prepared,
            mode=mode,
            task=task,
        )

        return {
            **prepared,
            **generation_payload,
        }

    def generate_from_text(
            self,
            text: str,
            mode: str = "general",
            task: str = "short_summary",
    ) -> Dict[str, Any]:

        prepared = self.process_text_for_mode(
            text=text,
            mode=mode,
            task=task,
        )

        generation_payload = self._generate_with_strategy(
            prepared=prepared,
            mode=mode,
            task=task,
        )

        return {
            **prepared,
            **generation_payload,
        }


if __name__ == "__main__":
    router = PipelineRouter()

    result = router.generate_from_file(
        file_path="sample1.pdf",
        mode="student",
        task="important_notes",
    )

    print(result["formatted_output"])