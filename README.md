---
title: Lumina AI Backend
emoji: 📄
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Lumina AI Backend

FastAPI backend for Lumina AI notes summarization, OCR, document analysis, legal pages, update metadata, and APK download routing.

## Hugging Face Space Setup

Use the Docker SDK. The included `Dockerfile` runs:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

Set these as Space secrets:

- `FIREBASE_SERVICE_ACCOUNT_JSON`
- `LUMINA_API_KEY`
- `LUMINA_GENERATION_PROVIDER=gemini`
- `LUMINA_MODEL_NAME=gemini-2.0-flash`
- `ALLOWED_ORIGINS=https://your-app-origin.example,https://your-site.example`

Optional:

- `APP_VERSION_NAME`
- `APP_VERSION_CODE`
- `APP_DOWNLOAD_PATH`
- `MAX_UPLOAD_BYTES`

## Memory Profile

The default deployment avoids loading heavyweight transformer and Paddle models at startup.

- RapidOCR ONNX is used as the deploy-friendly OCR model.
- Tesseract is installed in Docker as fallback OCR.
- `LUMINA_ENABLE_STRUCTURE_MODEL=false` by default.
- `LUMINA_ENABLE_PADDLEOCR=false` by default.

Only enable transformer/Paddle models on a larger CPU/GPU Space.
