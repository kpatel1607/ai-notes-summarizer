from pprint import pprint
from typing import Any, Dict

from model_systems.text_cleanup_pipeline import TextCleanupPipeline
from model_systems.smart_text_normalizer import SmartTextNormalizer
from model_systems.semantic_preservation_filter import SemanticPreservationFilter
from model_systems.document_structure_parser import DocumentStructureParser
from model_systems.semantic_chunker import SemanticChunker
from model_systems.mode_router import ModeRouter
from model_systems.prompt_builder import PromptBuilder
from model_systems.generation_service import GenerationService
from model_systems.response_postprocessor import ResponsePostprocessor
from model_systems.output_formatter import OutputFormatter


SAMPLE_TEXT = """
Telus Ahemdabad is HIRING !!! Role : Customer Service || Ticketing Process || Gaming Process Location: Fintech one, Floor 1, GIFT city, Gandhinagar, Gujarat What's in it for you? - Salary :: Up to 3- 4 LPA - Both side cabs (35 kms) -5 Days Working - Rotational Shifts and Rotational week-offs Who can apply? - Freshers can apply - People with strong communication & interpersonal skills - Strong Desktop gaming knowledge Contact only via WhatsApp message (Avoid Direct Calls) Pataliya Karan - 9408407446 Join For Upcoming Walk in drive and job Referal: https://Inkd.in/gb7 RiqjF #giftcity #ahmedabad #gandhinagar #Telus #telusdigital #fresher
""".strip()


def infer_structural_richness(structure: Dict[str, Any]) -> str:
    has_tables = bool(structure.get("tables"))
    has_key_values = bool(structure.get("key_value_fields"))
    has_forms = bool(structure.get("forms"))
    has_layout = bool(structure.get("layout_sections"))
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


def print_stage(
    title: str,
    text: str = "",
    metadata: Any = None,
) -> None:
    safe_text = str(text or "")

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    if safe_text:
        print(safe_text)
        print(
            f"\nWords: {len(safe_text.split())} | "
            f"Characters: {len(safe_text)}"
        )

    if metadata is not None:
        print("\nMetadata / object:")
        pprint(metadata, sort_dicts=False)


def main() -> None:
    # Change these values to test other combinations.
    mode = "professional"
    task = "table_format"
    user_plan = "free"

    cleanup_pipeline = TextCleanupPipeline()
    normalizer = SmartTextNormalizer()

    # Keep False for a deterministic first test.
    # Set True after sentence-transformers is installed.
    semantic_filter = SemanticPreservationFilter(
        enable_vectors=False,
    )

    structure_parser = DocumentStructureParser()
    semantic_chunker = SemanticChunker()
    mode_router = ModeRouter()
    prompt_builder = PromptBuilder()
    generation_service = GenerationService()
    response_postprocessor = ResponsePostprocessor()
    output_formatter = OutputFormatter()

    # ------------------------------------------------------------------
    # 1. Raw extracted OCR text
    # ------------------------------------------------------------------

    print_stage(
        "1. OCR EXTRACTED TEXT",
        SAMPLE_TEXT,
    )

    preliminary_structure = structure_parser.parse(
        SAMPLE_TEXT,
    )
    preliminary_richness = infer_structural_richness(
        preliminary_structure,
    )

    # ------------------------------------------------------------------
    # 2. Cleanup
    # ------------------------------------------------------------------

    cleaned = cleanup_pipeline.clean_ocr_text(
        text=SAMPLE_TEXT,
        mode=mode,
        task=task,
        structural_richness=preliminary_richness,
    )

    print_stage(
        "2. AFTER TEXT CLEANUP",
        cleaned["cleaned_text"],
        {
            "original_word_count":
                cleaned.get("original_word_count"),
            "cleaned_word_count":
                cleaned.get("cleaned_word_count"),
            "preservation_ratio":
                cleaned.get("preservation_ratio"),
            "cleanup_fallback_used":
                cleaned.get("cleanup_fallback_used"),
            "warnings":
                cleaned.get("warnings"),
            "cleanup_steps":
                cleaned.get("cleanup_steps"),
        },
    )

    # ------------------------------------------------------------------
    # 3. Normalization
    # ------------------------------------------------------------------

    normalized = normalizer.normalize(
        text=cleaned["cleaned_text"],
        mode=mode,
        task=task,
        structural_richness=preliminary_richness,
    )

    print_stage(
        "3. AFTER SMART NORMALIZATION",
        normalized["normalized_text"],
        {
            "original_word_count":
                normalized.get("original_word_count"),
            "normalized_word_count":
                normalized.get("normalized_word_count"),
            "preservation_ratio":
                normalized.get("preservation_ratio"),
            "normalization_fallback_used":
                normalized.get(
                    "normalization_fallback_used"
                ),
            "warnings":
                normalized.get("warnings"),
            "applied_fixes":
                normalized.get("applied_fixes"),
        },
    )

    # ------------------------------------------------------------------
    # 4. Semantic preservation
    # ------------------------------------------------------------------

    semantic_result = semantic_filter.filter_text(
        text=normalized["normalized_text"],
        mode=mode,
        task=task,
        structural_richness=preliminary_richness,
    )

    final_text = str(
        semantic_result.get(
            "filtered_text",
            normalized["normalized_text"],
        )
        or ""
    ).strip()

    print_stage(
        "4. AFTER SEMANTIC PRESERVATION",
        final_text,
        {
            "original_word_count":
                semantic_result.get("original_word_count"),
            "filtered_word_count":
                semantic_result.get("filtered_word_count"),
            "fallback_used":
                semantic_result.get("fallback_used"),
            "vector_used":
                semantic_result.get("vector_used"),
            "warnings":
                semantic_result.get("warnings"),
            "removed_lines":
                semantic_result.get("removed_lines"),
        },
    )

    # ------------------------------------------------------------------
    # 5. Structure and chunking
    # ------------------------------------------------------------------

    final_structure = structure_parser.parse(
        final_text,
    )
    final_richness = infer_structural_richness(
        final_structure,
    )

    chunks = semantic_chunker.chunk(
        pipeline_output={
            "final_text": final_text,
            "normalization": normalized,
            "semantic_preservation": semantic_result,
            "structure": final_structure,
            "understanding": {
                "metadata": {
                    "page_count": 1,
                    "word_count": len(final_text.split()),
                }
            },
        },
        max_words=350,
        max_chunks=16,
        user_plan=user_plan,
        mode=mode,
        task=task,
        structural_richness=final_richness,
        short_document_limit=500,
    )

    prompt_input = "\n\n".join(
        str(chunk.get("content", "") or "")
        for chunk in chunks.get("chunks", [])
    )

    print_stage(
        "5. EXACT TEXT AVAILABLE BEFORE MODE ROUTING",
        prompt_input,
        chunks.get("metadata", {}),
    )

    print_stage(
        "6. FINAL PARSED STRUCTURE",
        metadata=final_structure,
    )

    # Build a pipeline_output shaped like the real PipelineRouter output.
    pipeline_output = {
        "understanding": {
            "metadata": {
                "page_count": 1,
                "word_count": len(final_text.split()),
            }
        },
        "semantic_chunks": chunks,
        "extraction": {
            "text": SAMPLE_TEXT,
            "source": "manual_test",
            "confidence": 1.0,
        },
        "cleanup": cleaned,
        "normalization": normalized,
        "semantic_preservation": semantic_result,
        "structure": final_structure,
        "structural_richness": final_richness,
        "final_text": final_text,
        "routing": {
            "route_config": {
                "route": "manual_test",
                "model_tier": "standard",
            }
        },
    }

    # ------------------------------------------------------------------
    # 6. Mode routing
    # ------------------------------------------------------------------

    mode_output = mode_router.route(
        mode=mode,
        task=task,
        pipeline_output=pipeline_output,
    )

    # Ensure complete source data is explicitly available to PromptBuilder.
    # These assignments do not overwrite router fields.
    mode_output.setdefault(
        "source_text",
        final_text,
    )
    mode_output.setdefault(
        "semantic_chunks",
        chunks,
    )
    mode_output.setdefault(
        "structure",
        final_structure,
    )

    print_stage(
        "7. MODE ROUTER OUTPUT",
        metadata=mode_output,
    )

    # ------------------------------------------------------------------
    # 7. Prompt building
    # ------------------------------------------------------------------

    generation_prompt = prompt_builder.build(
        mode_output,
    )

    print_stage(
        "8. EXACT GENERATION PROMPT SENT TO MODEL",
        generation_prompt,
        {
            "mode": mode,
            "task": task,
            "source_word_count": len(final_text.split()),
            "prompt_word_count":
                len(generation_prompt.split()),
        },
    )

    # ------------------------------------------------------------------
    # 8. Model generation
    # ------------------------------------------------------------------

    max_tokens = generation_service.max_tokens_for_task(
        task=task,
        input_word_count=len(final_text.split()),
        strategy=chunks.get(
            "metadata",
            {},
        ).get(
            "recommended_strategy",
            "single_pass_generation",
        ),
    )

    generation_result = generation_service.generate(
        prompt=generation_prompt,
        max_tokens=max_tokens,
    )

    raw_generated_text = str(
        generation_result.get(
            "generated_text",
            "",
        )
        or ""
    ).strip()

    print_stage(
        "9. RAW MODEL OUTPUT",
        raw_generated_text,
        {
            "success":
                generation_result.get("success"),
            "provider":
                generation_result.get("provider"),
            "model":
                generation_result.get("model"),
            "error_type":
                generation_result.get("error_type"),
            "error":
                generation_result.get("error"),
            "max_tokens":
                max_tokens,
        },
    )

    if not generation_result.get("success"):
        print(
            "\nGeneration failed. Check LUMINA_API_KEY, "
            "LUMINA_API_URL, LUMINA_MODEL_NAME, and provider settings."
        )
        return

    # ------------------------------------------------------------------
    # 9. Postprocessing
    # ------------------------------------------------------------------

    postprocessed = response_postprocessor.process(
        generated_text=raw_generated_text,
        mode=mode,
        task=task,
    )

    processed_text = str(
        postprocessed.get(
            "processed_text",
            "",
        )
        or ""
    ).strip()

    print_stage(
        "10. AFTER RESPONSE POSTPROCESSING",
        processed_text,
        postprocessed,
    )

    # ------------------------------------------------------------------
    # 10. Output formatting
    # ------------------------------------------------------------------

    formatted_output = output_formatter.format(
        processed_text=processed_text,
        mode=mode,
        task=task,
        structure=final_structure,
        source_text=final_text,
        model=generation_result.get(
            "model",
            "",
        ),
        provider=generation_result.get(
            "provider",
            "",
        ),
        route_config={
            "route": "manual_test",
            "model_tier": "standard",
        },
        cached=False,
    )

    formatted_text = ""

    if isinstance(formatted_output, dict):
        formatted_text = str(
            formatted_output.get("markdown")
            or formatted_output.get("plain_text")
            or formatted_output.get("processed_text")
            or ""
        )
    else:
        formatted_text = str(formatted_output)

    print_stage(
        "11. FINAL FORMATTED OUTPUT",
        formatted_text,
        formatted_output,
    )

    # ------------------------------------------------------------------
    # Final diagnostic summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)
    print(
        "Compare these stages to find the first place where data disappears:"
    )
    print("5  -> complete text before mode routing")
    print("7  -> ModeRouter output")
    print("8  -> exact prompt")
    print("9  -> raw model output")
    print("10 -> postprocessed output")
    print("11 -> final formatter output")


if __name__ == "__main__":
    main()
