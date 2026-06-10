---
title: Lumina AI Backend
emoji: AI
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
- `PUBLIC_BASE_URL=https://lumina-ai.co.in`
- `CUSTOM_DOMAIN=https://lumina-ai.co.in`
- `ALLOWED_ORIGINS=https://lumina-ai.co.in,https://www.lumina-ai.co.in`
- `ALLOWED_HOSTS=lumina-ai.co.in,www.lumina-ai.co.in,*.hf.space,localhost,127.0.0.1`

Optional:

- `APP_VERSION_NAME`
- `APP_VERSION_CODE`
- `APP_DOWNLOAD_PATH`
- `MAX_UPLOAD_BYTES`

## Custom Domain Notes

Use `PUBLIC_BASE_URL` for the public API, download, and update URL that the Android app and policy pages should show. For this project, the main domain is `https://lumina-ai.co.in`. If Hugging Face does not let your Space use the apex domain directly, keep the Space on `*.hf.space` and put `lumina-ai.co.in`, `www.lumina-ai.co.in`, or an API subdomain such as `api.lumina-ai.co.in` in front of it with a reverse proxy such as Cloudflare. The backend is prepared for any of those options as long as `PUBLIC_BASE_URL`, `ALLOWED_ORIGINS`, and `ALLOWED_HOSTS` include the public domain.

For Play Store, use your own website domain for the privacy policy, support, and account deletion URLs. The app API can still be hosted on Hugging Face behind your API subdomain.

## Memory Profile

The default deployment avoids loading heavyweight transformer and Paddle models at startup.

- RapidOCR ONNX is used as the deploy-friendly OCR model.
- Tesseract is installed in Docker as fallback OCR.
- `LUMINA_ENABLE_STRUCTURE_MODEL=false` by default.
- `LUMINA_ENABLE_PADDLEOCR=false` by default.

Only enable transformer/Paddle models on a larger CPU/GPU Space.

## Prompt Routing

Short pasted notes use a compact prompt path. Larger PDFs, table-heavy files, and chunk-limited documents still use the structured prompt path.
