from model_systems.pipeline_router import (
    PipelineRouter,
)

router = PipelineRouter()

# TEXT TEST
text_result = router.process_text(
    "Chapter 1: Machine Learning. Define supervised learning and explain classification."
)

print("\nTEXT RESULT:")
print(text_result)

# IMAGE TEST
image_result = router.process_file(
    "sample.png"
)

print("\nIMAGE RESULT:")
print(image_result)

# PDF TEST
pdf_result = router.process_file(
    "sample1.pdf"
)

print("\nPDF RESULT:")
print(pdf_result)