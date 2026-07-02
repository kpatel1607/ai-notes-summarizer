from typing import Any, Dict, List

from model_systems.input.input_understanding import (
    InputUnderstandingSystem,
)
from model_systems.input.input_analyzer import (
    DocumentInputAnalyzer,
)
from model_systems.input.feature_extractor import (
    FeatureExtractor,
)
from model_systems.input.complexity_scorer import (
    ComplexityScorer,
)
from model_systems.core.route_selector import (
    RouteSelector,
)
from model_systems.extraction.ocr_pipeline import (
    OCRPipeline,
)
from model_systems.preprocessing.text_cleanup_pipeline import (
    TextCleanupPipeline,
)
from model_systems.structure.document_structure_parser import (
    DocumentStructureParser,
)
from model_systems.core.mode_router import (
    ModeRouter,
)
from model_systems.generation.prompt_builder import (
    PromptBuilder,
)
from model_systems.retrieval.semantic_chunker import (
    SemanticChunker,
)
from model_systems.generation.generation_service import (
    GenerationService,
)
from model_systems.preprocessing.smart_text_normalizer import (
    SmartTextNormalizer,
)
from model_systems.preprocessing.semantic_preservation_filter import (
    SemanticPreservationFilter,
)
from model_systems.generation.response_postprocessor import (
    ResponsePostprocessor,
)
from model_systems.generation.output_formatter import (
    OutputFormatter,
)


class PipelineRouter:
    """
    Main orchestration layer for Lumina's document-processing pipeline.

    Pipeline:
        input analysis
        -> extraction/OCR
        -> preliminary structure detection
        -> conservative cleanup
        -> safe normalization
        -> semantic preservation
        -> final structure parsing
        -> semantic chunking
        -> routing
        -> prompt building
        -> generation
        -> postprocessing
        -> output formatting
    """

    def __init__(self) -> None:
        self.input_system = InputUnderstandingSystem()
        self.input_analyzer = DocumentInputAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.complexity_scorer = ComplexityScorer()
        self.route_selector = RouteSelector()
        self.ocr_pipeline = OCRPipeline()
        self.cleanup_pipeline = TextCleanupPipeline()
        self.structure_parser = DocumentStructureParser()
        self.mode_router = ModeRouter()
        self.prompt_builder = PromptBuilder()
        self.semantic_chunker = SemanticChunker()
        self.generation_service = GenerationService()
        self.smart_normalizer = SmartTextNormalizer()

        # The semantic-preservation implementation should lazy-load its
        # embedding model on first use and then reuse it.
        self.semantic_preservation_filter = (
            SemanticPreservationFilter(
                enable_vectors=True,
            )
        )

        self.response_postprocessor = (
            ResponsePostprocessor()
        )
        self.output_formatter = OutputFormatter()

    # ------------------------------------------------------------------
    # Public processing methods
    # ------------------------------------------------------------------

    def process_text(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        mode = self._normalize_mode(mode)
        task = self._normalize_task(task)
        user_plan = self._normalize_user_plan(user_plan)

        raw_text = str(text or "")

        input_analysis = self.input_analyzer.analyze_text(
            raw_text,
        )
        understanding = self.input_system.analyze_text(
            raw_text,
        )
        extracted = self.ocr_pipeline.extract_from_text(
            raw_text,
        )

        extracted_text = str(
            extracted.get("text", "") or ""
        ).strip()

        if not extracted_text:
            return self._build_empty_pipeline_output(
                input_analysis=input_analysis,
                understanding=understanding,
                extracted=extracted,
                mode=mode,
                task=task,
                user_plan=user_plan,
                text=raw_text,
            )

        preliminary_structure = self.structure_parser.parse(
            extracted_text,
        )
        preliminary_richness = (
            self._infer_structural_richness(
                preliminary_structure,
            )
        )

        cleaned = self.cleanup_pipeline.clean_ocr_text(
            text=extracted_text,
            mode=mode,
            task=task,
            structural_richness=preliminary_richness,
        )

        normalized = self.smart_normalizer.normalize(
            text=cleaned["cleaned_text"],
            mode=mode,
            task=task,
            structural_richness=preliminary_richness,
        )

        semantic_preserved = (
            self.semantic_preservation_filter.filter_text(
                text=normalized["normalized_text"],
                mode=mode,
                task=task,
                structural_richness=preliminary_richness,
            )
        )

        final_text = str(
            semantic_preserved.get(
                "filtered_text",
                normalized["normalized_text"],
            )
            or ""
        ).strip()

        structure = self.structure_parser.parse(
            final_text,
        )
        structural_richness = (
            self._infer_structural_richness(
                structure,
            )
        )

        semantic_chunks = self.semantic_chunker.chunk(
            pipeline_output={
                "final_text": final_text,
                "normalization": normalized,
                "semantic_preservation":
                    semantic_preserved,
                "structure": structure,
                "understanding":
                    understanding.__dict__,
            },
            max_words=350,
            max_chunks=16,
            user_plan=user_plan,
            mode=mode,
            task=task,
            structural_richness=
                structural_richness,
            short_document_limit=500,
        )

        pipeline_output = {
            "understanding": understanding.__dict__,
            "semantic_chunks": semantic_chunks,
            "extraction": extracted,
            "cleanup": cleaned,
            "normalization": normalized,
            "semantic_preservation":
                semantic_preserved,
            "structure": structure,
            "structural_richness":
                structural_richness,
            "final_text": final_text,
        }

        pipeline_output["routing"] = (
            self._build_route_decision(
                text=raw_text,
                input_analysis=input_analysis,
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                user_plan=user_plan,
            )
        )

        return pipeline_output

    def process_file(
        self,
        file_path: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        mode = self._normalize_mode(mode)
        task = self._normalize_task(task)
        user_plan = self._normalize_user_plan(user_plan)

        input_analysis = self.input_analyzer.analyze_file(
            file_path,
        )
        understanding = self.input_system.analyze_file(
            file_path,
        )

        strategy = understanding.extraction_strategy

        preliminary_features = (
            self.feature_extractor.extract(
                file_path=file_path,
                input_analysis=input_analysis,
                mode=mode,
                task_type=task,
                user_plan=user_plan,
            )
        )

        preliminary_complexity = (
            self.complexity_scorer.score(
                preliminary_features,
            )
        )

        preliminary_route = (
            self.route_selector.select(
                features=preliminary_features,
                complexity=preliminary_complexity,
            )
        )

        layout_structure = None

        if input_analysis.input_type in {
            "digital_pdf",
            "scanned_pdf",
            "mixed_document",
        }:
            extracted = (
                self.ocr_pipeline.extract_from_pdf(
                    file_path,
                    route_config=preliminary_route,
                    features=preliminary_features,
                    mode=mode,
                    task=task,
                    user_plan=user_plan,
                )
            )

        elif strategy in {
            "image_ocr",
            "enhanced_image_ocr",
            "image_preprocessing_then_ocr",
        }:
            extracted = (
                self.ocr_pipeline.extract_from_image(
                    file_path,
                )
            )

            layout_structure = (
                self.structure_parser
                .parse_document_image(
                    file_path,
                )
            )

        else:
            extracted = {
                "text": "",
                "source": "unsupported",
                "confidence": 0.0,
            }

        extracted_text = str(
            extracted.get("text", "") or ""
        ).strip()

        if not extracted_text:
            return self._build_empty_pipeline_output(
                input_analysis=input_analysis,
                understanding=understanding,
                extracted=extracted,
                mode=mode,
                task=task,
                user_plan=user_plan,
                file_path=file_path,
            )

        preliminary_text_structure = (
            self.structure_parser.parse(
                extracted_text,
            )
        )

        preliminary_richness = (
            self._infer_structural_richness(
                preliminary_text_structure,
            )
        )

        cleaned = self.cleanup_pipeline.clean_ocr_text(
            text=extracted_text,
            mode=mode,
            task=task,
            structural_richness=preliminary_richness,
        )

        normalized = self.smart_normalizer.normalize(
            text=cleaned["cleaned_text"],
            mode=mode,
            task=task,
            structural_richness=preliminary_richness,
        )

        semantic_preserved = (
            self.semantic_preservation_filter.filter_text(
                text=normalized["normalized_text"],
                mode=mode,
                task=task,
                structural_richness=preliminary_richness,
            )
        )

        final_text = str(
            semantic_preserved.get(
                "filtered_text",
                normalized["normalized_text"],
            )
            or ""
        ).strip()

        text_structure = self.structure_parser.parse(
            final_text,
        )

        extracted_tables = (
            extracted.get("tables", []) or []
        )

        if extracted_tables:
            text_structure["tables"] = [
                *text_structure.get(
                    "tables",
                    [],
                ),
                *[
                    {
                        "source":
                            "pdf_extractor",
                        "rows": table,
                    }
                    for table in extracted_tables
                ],
            ]

            text_structure["metadata"] = {
                **text_structure.get(
                    "metadata",
                    {},
                ),
                "table_count": len(
                    text_structure["tables"]
                ),
            }

        structure = (
            self.structure_parser
            .merge_text_and_layout_structure(
                text_structure=text_structure,
                layout_structure=layout_structure,
            )
        )

        structural_richness = (
            self._infer_structural_richness(
                structure,
            )
        )

        semantic_chunks = self.semantic_chunker.chunk(
            pipeline_output={
                "final_text": final_text,
                "normalization": normalized,
                "semantic_preservation":
                    semantic_preserved,
                "structure": structure,
                "understanding":
                    understanding.__dict__,
            },
            max_words=350,
            max_chunks=16,
            user_plan=user_plan,
            mode=mode,
            task=task,
            structural_richness=
                structural_richness,
            short_document_limit=500,
        )

        pipeline_output = {
            "understanding": understanding.__dict__,
            "semantic_chunks": semantic_chunks,
            "extraction": extracted,
            "cleanup": cleaned,
            "normalization": normalized,
            "semantic_preservation":
                semantic_preserved,
            "structure": structure,
            "structural_richness":
                structural_richness,
            "final_text": final_text,
        }

        pipeline_output["routing"] = (
            self._build_route_decision(
                file_path=file_path,
                input_analysis=input_analysis,
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                user_plan=user_plan,
            )
        )

        return pipeline_output

    # ------------------------------------------------------------------
    # Structural analysis and routing
    # ------------------------------------------------------------------

    def _infer_structural_richness(
        self,
        structure: Dict[str, Any],
    ) -> str:
        structure = structure or {}

        has_tables = bool(
            structure.get("tables")
        )
        has_key_values = bool(
            structure.get("key_value_fields")
        )
        has_forms = bool(
            structure.get("forms")
        )
        has_layout = bool(
            structure.get("layout_sections")
        )
        has_lists = bool(
            structure.get("numbered_items")
            or structure.get("roman_items")
        )
        has_sections = bool(
            structure.get("sections")
            or structure.get("paragraphs")
        )

        active_types = sum(
            [
                has_tables,
                has_key_values,
                has_forms,
                has_layout,
                has_lists,
                has_sections,
            ]
        )

        if active_types >= 3:
            return "mixed"

        if has_forms:
            return "form"

        if has_tables:
            return "table"

        if has_key_values:
            return "key_value"

        if has_layout:
            return "multi_column"

        if has_lists:
            return "list"

        if has_sections:
            return "rich"

        return "plain"

    def _build_route_decision(
        self,
        input_analysis: Any,
        pipeline_output: Dict[str, Any],
        mode: str,
        task: str,
        user_plan: str,
        text: str = "",
        file_path: str = "",
    ) -> Dict[str, Any]:
        features = self.feature_extractor.extract(
            text=text,
            file_path=file_path,
            input_analysis=input_analysis,
            pipeline_output=pipeline_output,
            mode=mode,
            task_type=task,
            user_plan=user_plan,
        )

        complexity = self.complexity_scorer.score(
            features,
        )

        route_config = self.route_selector.select(
            features=features,
            complexity=complexity,
        )

        return {
            "input_analysis":
                input_analysis.__dict__,
            "features": features,
            "complexity": complexity,
            "route_config": route_config,
        }

    def test_layout_parser(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        layout_structure = (
            self.structure_parser
            .parse_document_image(
                image_path,
            )
        )

        return {
            "file_path": image_path,
            "layout_structure":
                layout_structure,
        }

    # ------------------------------------------------------------------
    # Mode preparation
    # ------------------------------------------------------------------

    def process_text_for_mode(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        mode = self._normalize_mode(mode)
        task = self._normalize_task(task)
        user_plan = self._normalize_user_plan(user_plan)

        pipeline_output = self.process_text(
            text=text,
            mode=mode,
            task=task,
            user_plan=user_plan,
        )

        mode_output = self.mode_router.route(
            mode=mode,
            task=task,
            pipeline_output=pipeline_output,
        )

        generation_prompt = (
            self.prompt_builder.build(
                mode_output,
            )
        )

        return {
            "pipeline_output": pipeline_output,
            "mode_output": mode_output,
            "generation_prompt":
                generation_prompt,
        }

    def process_file_for_mode(
        self,
        file_path: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        mode = self._normalize_mode(mode)
        task = self._normalize_task(task)
        user_plan = self._normalize_user_plan(user_plan)

        pipeline_output = self.process_file(
            file_path=file_path,
            mode=mode,
            task=task,
            user_plan=user_plan,
        )

        mode_output = self.mode_router.route(
            mode=mode,
            task=task,
            pipeline_output=pipeline_output,
        )

        generation_prompt = (
            self.prompt_builder.build(
                mode_output,
            )
        )

        return {
            "pipeline_output": pipeline_output,
            "mode_output": mode_output,
            "generation_prompt":
                generation_prompt,
        }

    # ------------------------------------------------------------------
    # Generation strategy
    # ------------------------------------------------------------------

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
            return (
                self._generate_hierarchical_summary(
                    prepared=prepared,
                    mode=mode,
                    task=task,
                )
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
        pipeline_output = prepared.get(
            "pipeline_output",
            {},
        )

        route_config = (
            pipeline_output.get(
                "routing",
                {},
            )
            .get(
                "route_config",
                {},
            )
        )

        source_text = str(
            pipeline_output.get(
                "final_text",
                "",
            )
            or ""
        )

        generation_result = (
            self.generation_service.generate(
                prepared.get(
                    "generation_prompt",
                    "",
                ),
                max_tokens=(
                    self.generation_service
                    .max_tokens_for_task(
                        task=task,
                        input_word_count=len(
                            source_text.split()
                        ),
                        strategy=
                            "single_pass_generation",
                    )
                ),
            )
        )

        if not generation_result.get(
            "success",
        ):
            generation_result[
                "generated_text"
            ] = self._safe_fallback_output(
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                error=generation_result.get(
                    "error",
                    "",
                ),
            )

        postprocessed = (
            self.response_postprocessor.process(
                generated_text=
                    generation_result.get(
                        "generated_text",
                        "",
                    ),
                mode=mode,
                task=task,
            )
        )

        formatted_output = (
            self.output_formatter.format(
                processed_text=postprocessed[
                    "processed_text"
                ],
                mode=mode,
                task=task,
                structure=pipeline_output.get(
                    "structure",
                    {},
                ),
                source_text=source_text,
                model=generation_result.get(
                    "model",
                    "",
                ),
                provider=
                    generation_result.get(
                        "provider",
                        "",
                    ),
                route_config=route_config,
                cached=False,
            )
        )

        return {
            "generation_strategy":
                "single_pass_generation",
            "generation_result":
                generation_result,
            "postprocessed_output":
                postprocessed,
            "formatted_output":
                formatted_output,
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

        route_config = (
            pipeline_output.get(
                "routing",
                {},
            )
            .get(
                "route_config",
                {},
            )
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

        chunk_summaries: List[
            Dict[str, Any]
        ] = []

        for index, chunk in enumerate(chunks):
            chunk_prompt = (
                self._build_chunk_summary_prompt(
                    chunk=chunk,
                    index=index,
                    mode=mode,
                    task=task,
                )
            )

            chunk_word_count = len(
                str(
                    chunk.get(
                        "content",
                        "",
                    )
                    or ""
                ).split()
            )

            chunk_token_budget = min(
                max(
                    900,
                    int(chunk_word_count * 2.2),
                ),
                2200,
            )

            chunk_generation = self.generation_service.generate(
                chunk_prompt,
                max_tokens=chunk_token_budget,
            )

            generated_text = str(
                chunk_generation.get(
                    "generated_text",
                    "",
                )
                or ""
            ).strip()

            if (
                not chunk_generation.get(
                    "success",
                )
                or not generated_text
            ):
                continue

            chunk_summaries.append(
                {
                    "chunk_index": index,
                    "chunk_type":
                        chunk.get(
                            "chunk_type",
                            "",
                        ),
                    "title":
                        chunk.get(
                            "title",
                            "",
                        ),
                    "success": True,
                    "summary": generated_text,
                    "provider":
                        chunk_generation.get(
                            "provider",
                            "",
                        ),
                    "model":
                        chunk_generation.get(
                            "model",
                            "",
                        ),
                }
            )

        if not chunk_summaries:
            return (
                self._build_hierarchical_fallback(
                    pipeline_output=
                        pipeline_output,
                    route_config=route_config,
                    metadata=metadata,
                    mode=mode,
                    task=task,
                    error=(
                        "All hierarchical chunk "
                        "generations failed."
                    ),
                )
            )

        merge_prompt = (
            self._build_merge_summary_prompt(
                chunk_summaries=chunk_summaries,
                metadata=metadata,
                mode=mode,
                task=task,
            )
        )

        final_generation = (
            self.generation_service.generate(
                merge_prompt,
                max_tokens=(
                    self.generation_service
                    .max_tokens_for_task(
                        task=task,
                        input_word_count=sum(
                            len(
                                str(
                                    item.get(
                                        "summary",
                                        "",
                                    )
                                ).split()
                            )
                            for item
                            in chunk_summaries
                        ),
                        strategy=
                            "hierarchical_summary",
                    )
                ),
            )
        )

        if not final_generation.get(
            "success",
        ):
            final_generation[
                "generated_text"
            ] = self._safe_fallback_output(
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                error=final_generation.get(
                    "error",
                    "",
                ),
            )

        postprocessed = (
            self.response_postprocessor.process(
                generated_text=
                    final_generation.get(
                        "generated_text",
                        "",
                    ),
                mode=mode,
                task=task,
            )
        )

        formatted_output = (
            self.output_formatter.format(
                processed_text=postprocessed[
                    "processed_text"
                ],
                mode=mode,
                task=task,
                structure=pipeline_output.get(
                    "structure",
                    {},
                ),
                source_text=
                    pipeline_output.get(
                        "final_text",
                        "",
                    ),
                model=
                    final_generation.get(
                        "model",
                        "",
                    ),
                provider=
                    final_generation.get(
                        "provider",
                        "",
                    ),
                route_config=route_config,
                cached=False,
            )
        )

        return {
            "generation_strategy":
                "hierarchical_summary",
            "chunk_summaries":
                chunk_summaries,
            "generation_result":
                final_generation,
            "postprocessed_output":
                postprocessed,
            "formatted_output":
                formatted_output,
            "free_tier_limit_response":
                self._build_limit_response(
                    metadata,
                ),
        }

    def _build_hierarchical_fallback(
        self,
        pipeline_output: Dict[str, Any],
        route_config: Dict[str, Any],
        metadata: Dict[str, Any],
        mode: str,
        task: str,
        error: str,
    ) -> Dict[str, Any]:
        fallback_text = (
            self._safe_fallback_output(
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                error=error,
            )
        )

        postprocessed = (
            self.response_postprocessor.process(
                generated_text=fallback_text,
                mode=mode,
                task=task,
            )
        )

        formatted_output = (
            self.output_formatter.format(
                processed_text=postprocessed[
                    "processed_text"
                ],
                mode=mode,
                task=task,
                structure=pipeline_output.get(
                    "structure",
                    {},
                ),
                source_text=
                    pipeline_output.get(
                        "final_text",
                        "",
                    ),
                model="",
                provider="fallback",
                route_config=route_config,
                cached=False,
            )
        )

        return {
            "generation_strategy":
                "hierarchical_summary",
            "chunk_summaries": [],
            "generation_result": {
                "success": False,
                "generated_text":
                    fallback_text,
                "error": error,
                "provider": "fallback",
                "model": "",
            },
            "postprocessed_output":
                postprocessed,
            "formatted_output":
                formatted_output,
            "free_tier_limit_response":
                self._build_limit_response(
                    metadata,
                ),
        }

    # ------------------------------------------------------------------
    # Hierarchical prompts
    # ------------------------------------------------------------------

    def _build_chunk_summary_prompt(
        self,
        chunk: Dict[str, Any],
        index: int,
        mode: str,
        task: str,
    ) -> str:
        return f"""
You are Lumina AI.

Create a faithful intermediate representation of this document chunk for
later completion of the user's final task.

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
- Preserve all information relevant to the final task.
- Preserve facts, definitions, claims, examples, requirements, decisions,
  actions, dates, numbers, names, formulas, and instructions.
- Do not add outside information.
- Do not discard short but meaningful fields.
- Avoid stylistic rewriting at this stage.
- Keep the intermediate representation concise but information-complete.

Chunk content:
{chunk.get("content", "")}
""".strip()

    def _build_merge_summary_prompt(
        self,
        chunk_summaries: List[
            Dict[str, Any]
        ],
        metadata: Dict[str, Any],
        mode: str,
        task: str,
    ) -> str:
        summaries_text = "\n\n".join(
            [
                (
                    f"Chunk "
                    f"{item['chunk_index'] + 1}\n"
                    f"Title: "
                    f"{item.get('title', '')}\n"
                    f"Intermediate content:\n"
                    f"{item.get('summary', '')}"
                )
                for item in chunk_summaries
                if item.get("summary")
            ]
        )

        limit_note = ""

        if metadata.get(
            "limit_applied",
        ):
            limit_note = """
Important limitation:
Only part of the document was processed because the current chunk limit was
reached. Clearly mention that the output is based on the processed portion.
""".strip()

        return f"""
You are Lumina AI.

Create the final requested output from the intermediate chunk representations.

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
- Use only the intermediate content below.
- Do not add unsupported facts.
- Preserve important details needed for the requested task.
- Avoid repetition.
- Match the requested mode and task exactly.
- Produce a complete, clean, and properly structured final output.

Intermediate chunk content:
{summaries_text}
""".strip()

    # ------------------------------------------------------------------
    # Fallbacks and limits
    # ------------------------------------------------------------------

    def _build_limit_response(
        self,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        limit_applied = bool(
            metadata.get(
                "limit_applied",
                False,
            )
        )

        if not limit_applied:
            return {
                "limit_applied": False,
                "message": "",
            }

        return {
            "limit_applied": True,
            "message": (
                "This output is based on the processed "
                "portion of the document because the "
                "current chunk limit was reached."
            ),
            "chunks_used": metadata.get(
                "chunk_count",
                0,
            ),
            "original_chunks":
                metadata.get(
                    "original_chunk_count",
                    0,
                ),
            "user_plan": metadata.get(
                "user_plan",
                "free",
            ),
        }

    def _safe_fallback_output(
        self,
        pipeline_output: Dict[str, Any],
        mode: str,
        task: str,
        error: str = "",
    ) -> str:
        source_text = str(
            pipeline_output.get(
                "final_text",
                "",
            )
            or ""
        ).strip()

        if not source_text:
            return (
                "Generation unavailable\n\n"
                "The document text could not be "
                "extracted clearly enough to "
                "generate this output."
            )

        preview = " ".join(
            source_text.split()[:220]
        )

        if task == "email_draft":
            return (
                "Subject: Follow-up regarding the "
                "provided document\n\n"
                "Hello,\n\n"
                "I am writing regarding the "
                "information contained in the "
                "provided document. The AI "
                "generation service is temporarily "
                "unavailable, so the complete email "
                "draft could not be prepared.\n\n"
                "Relevant extracted information:\n"
                f"{preview}\n\n"
                "Regards,\n"
                "[Your Name]"
            )

        if task == "table_format":
            lines = [
                line.strip()
                for line in source_text.splitlines()
                if line.strip()
            ][:20]

            rows = [
                "| Item | Details |",
                "| --- | --- |",
            ]

            for index, line in enumerate(
                lines,
                start=1,
            ):
                safe_line = line.replace(
                    "|",
                    "/",
                )
                rows.append(
                    f"| {index} | {safe_line} |"
                )

            return "\n".join(rows)

        heading = (
            "Generation temporarily unavailable"
        )

        if mode == "student":
            heading = (
                "Study Output Temporarily Limited"
            )
        elif mode == "professional":
            heading = (
                "Professional Output "
                "Temporarily Limited"
            )

        error_note = (
            f"\n\nTechnical note: {error}"
            if error
            else ""
        )

        return (
            f"{heading}\n\n"
            "The AI provider did not return a "
            "complete response, so Lumina preserved "
            "the extracted source content instead "
            "of returning an empty document.\n\n"
            f"{preview}"
            f"{error_note}"
        )

    def _build_empty_pipeline_output(
        self,
        input_analysis: Any,
        understanding: Any,
        extracted: Dict[str, Any],
        mode: str,
        task: str,
        user_plan: str,
        text: str = "",
        file_path: str = "",
    ) -> Dict[str, Any]:
        semantic_chunks = {
            "chunks": [],
            "metadata": {
                "chunk_count": 0,
                "original_chunk_count": 0,
                "word_count": 0,
                "page_count": 0,
                "limit_applied": False,
                "chunk_limit_required": False,
                "recommended_strategy":
                    "single_pass_generation",
                "warnings": [
                    "no_extractable_text",
                ],
            },
        }

        pipeline_output = {
            "understanding":
                understanding.__dict__,
            "semantic_chunks":
                semantic_chunks,
            "extraction": extracted,
            "cleanup": {},
            "normalization": {},
            "semantic_preservation": {},
            "structure": {},
            "structural_richness": "plain",
            "final_text": "",
        }

        pipeline_output["routing"] = (
            self._build_route_decision(
                text=text,
                file_path=file_path,
                input_analysis=input_analysis,
                pipeline_output=pipeline_output,
                mode=mode,
                task=task,
                user_plan=user_plan,
            )
        )

        return pipeline_output

    # ------------------------------------------------------------------
    # Final public generation methods
    # ------------------------------------------------------------------

    def generate_prepared(
        self,
        prepared: Dict[str, Any],
        mode: str = "general",
        task: str = "short_summary",
    ) -> Dict[str, Any]:
        mode = self._normalize_mode(mode)
        task = self._normalize_task(task)

        generation_payload = (
            self._generate_with_strategy(
                prepared=prepared,
                mode=mode,
                task=task,
            )
        )

        return {
            **prepared,
            **generation_payload,
        }

    def generate_from_file(
        self,
        file_path: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        prepared = self.process_file_for_mode(
            file_path=file_path,
            mode=mode,
            task=task,
            user_plan=user_plan,
        )

        return self.generate_prepared(
            prepared=prepared,
            mode=mode,
            task=task,
        )

    def generate_from_text(
        self,
        text: str,
        mode: str = "general",
        task: str = "short_summary",
        user_plan: str = "free",
    ) -> Dict[str, Any]:
        prepared = self.process_text_for_mode(
            text=text,
            mode=mode,
            task=task,
            user_plan=user_plan,
        )

        return self.generate_prepared(
            prepared=prepared,
            mode=mode,
            task=task,
        )

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    def _normalize_mode(
        self,
        mode: str,
    ) -> str:
        normalized = (
            mode or "general"
        ).strip().lower()

        if normalized not in {
            "general",
            "student",
            "professional",
        }:
            return "general"

        return normalized

    def _normalize_task(
        self,
        task: str,
    ) -> str:
        normalized = (
            task or "short_summary"
        ).strip().lower()

        return normalized or "short_summary"

    def _normalize_user_plan(
        self,
        user_plan: str,
    ) -> str:
        normalized = (
            user_plan or "free"
        ).strip().lower()

        if normalized not in {
            "free",
            "pro",
            "premium",
        }:
            return "free"

        return normalized


if __name__ == "__main__":
    router = PipelineRouter()

    result = router.generate_from_file(
        file_path="sample1.pdf",
        mode="student",
        task="important_notes",
        user_plan="free",
    )

    print(result.get("formatted_output"))
