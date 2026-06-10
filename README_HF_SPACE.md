---
title: Lumina AI Backend
emoji: AI
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
- `PUBLIC_BASE_URL=https://lumina-ai.co.in`
- `CUSTOM_DOMAIN=https://lumina-ai.co.in`
- `ALLOWED_ORIGINS=https://lumina-ai.co.in,https://www.lumina-ai.co.in`
- `ALLOWED_HOSTS=lumina-ai.co.in,www.lumina-ai.co.in,*.hf.space,localhost,127.0.0.1`

## Domain Notes

For Play Store, publish privacy, terms, support, and account deletion links on `https://lumina-ai.co.in`. The API can run on Hugging Face and still expose update/download links using `PUBLIC_BASE_URL`.

If Hugging Face custom domain setup is not available for your Space, use a reverse proxy such as Cloudflare in front of the `*.hf.space` URL and set `PUBLIC_BASE_URL` to `https://lumina-ai.co.in`, `https://www.lumina-ai.co.in`, or an API subdomain such as `https://api.lumina-ai.co.in`.

## Memory Notes

The default Space setup avoids loading large transformer and Paddle models at startup.

- `rapidocr-onnxruntime` is used for deploy-friendly OCR.
- Tesseract is installed in the Docker image as fallback.
- `LUMINA_ENABLE_STRUCTURE_MODEL=false` by default.
- `LUMINA_ENABLE_PADDLEOCR=false` by default.

Only enable heavier models on a larger CPU/GPU Space.
