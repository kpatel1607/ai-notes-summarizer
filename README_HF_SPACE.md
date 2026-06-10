---
title: Lumina AI Backend
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Lumina AI Backend

FastAPI backend for Lumina AI notes summarization, OCR, document analysis, and policy/download pages.

## Required Secrets

Set these in Hugging Face Space secrets:

- `FIREBASE_SERVICE_ACCOUNT_JSON`
- `LUMINA_API_KEY`
- `LUMINA_GENERATION_PROVIDER=gemini`
- `LUMINA_MODEL_NAME=gemini-2.0-flash`
- `BASE_URL` or `APP_DOWNLOAD_PATH` if using a custom domain/download URL
- `ALLOWED_ORIGINS` with your app and website origins

## Memory Notes

The default Space setup avoids loading large transformer and Paddle models at startup.

- `rapidocr-onnxruntime` is used for deploy-friendly OCR.
- Tesseract is installed in the Docker image as fallback.
- `LUMINA_ENABLE_STRUCTURE_MODEL=false` by default.
- `LUMINA_ENABLE_PADDLEOCR=false` by default.

Only enable heavier models on a larger CPU/GPU Space.
