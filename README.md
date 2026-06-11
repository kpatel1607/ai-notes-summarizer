---
title: Lumina AI Backend
emoji: AI
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Lumina AI Notes Summarizer

Lumina AI is an AI notes and document summarizer made of a FastAPI backend, a Flutter Android app, Firebase services, a public policy/download website, and a Cloudflare Worker reverse proxy. The app lets users paste notes, extract text from PDFs/images/camera scans, choose an output mode, generate structured AI documents, save results to cloud history, organize them into folders, and download future app updates from the official domain.

The production public domain is:

```text
https://lumina-ai.co.in
```

The current Hugging Face Space origin is:

```text
https://kpatel1607-lumina.hf.space
```

Cloudflare is used as the public reverse proxy so users and Play Store policy links can use the Lumina domain while the backend continues running on Hugging Face Spaces.

## Repository Layout

This workspace currently has three important project folders:

```text
ai_notes_summariser/
  ai_backend/              FastAPI backend, model pipeline, website, APK, Cloudflare Worker
  notes_summarizer_app/    Flutter app source for Android and other Flutter targets
  hf_lumina_deploy/        Hugging Face deployment copy/snapshot
```

`ai_backend` is the active Git repository that is pushed to GitHub and deployed to Hugging Face. The Flutter app is a sibling folder, not part of the backend Git repository, so app source changes are local unless they are copied, committed, or moved into the repo later.

## High-Level Architecture

```text
Flutter Android App
  - Firebase Authentication
  - On-device PDF/image/camera extraction
  - Sends text to backend generation API
  - Saves generated summaries/folders/history to Firestore
  - Checks /app-version for update prompts

Cloudflare Worker
  - Public endpoint at lumina-ai.co.in
  - Reverse proxies requests to Hugging Face Space
  - Adds CORS, timeout handling, upload size guard, and security headers
  - Prevents stale APK caching on /download-apk

FastAPI Backend
  - Serves website, legal pages, update metadata, and APK download
  - Verifies Firebase bearer tokens for generation APIs
  - Applies rate limits and daily usage counters
  - Runs OCR, cleanup, structure parsing, semantic chunking, prompt building, generation, and formatting

Firebase
  - Authentication
  - Firestore user profiles, summaries, folders, favorites, usage counters, feedback
  - Analytics, Crashlytics, and App Check support in the app
```

## Backend

The backend is in `ai_backend` and is built with FastAPI.

Important files:

```text
main.py                         FastAPI app, API routes, public website, legal pages
requirements.txt                Python dependencies
Dockerfile                      Hugging Face Docker Space runtime
runtime.txt                     Python runtime declaration
static/Lumina-AI.apk            Public Android APK served by /download-apk
model_systems/                  OCR, document analysis, prompt, generation, and formatting pipeline
cloudflare-worker/              Cloudflare Worker reverse proxy
```

### Backend Responsibilities

- Serve the public website at `/`.
- Serve Play Store-ready legal pages:
  - `/privacy-policy`
  - `/terms-and-conditions`
- Serve Android update metadata at `/app-version`.
- Serve the latest APK at `/download-apk`.
- Serve the download landing page at `/download-app` and `/update`.
- Verify Firebase authentication for protected generation endpoints.
- Enforce request rate limits and per-user daily usage limits.
- Extract, clean, normalize, structure, chunk, summarize, and format user content.

## Public Website

The backend renders a lightweight professional website directly from `main.py`.

Current public pages:

```text
GET /                         Homepage
GET /download-app             Android download page
GET /update                   Alias for the download page
GET /privacy-policy           Privacy Policy
GET /terms-and-conditions     Terms & Conditions
GET /app-version              JSON metadata for app updates
GET /download-apk             APK binary download
GET /health                   Backend health/status JSON
```

The website includes animated UI preview elements, app status loading, APK version display, mode previews, policy links, and download links. Legal pages use a simple top Back button so the navigation is cleaner when reading policy content.

## Android Update Flow

The Flutter app uses `AppUpdateService` to call:

```text
https://lumina-ai.co.in/app-version
```

The backend returns metadata such as:

```json
{
  "latestVersionName": "2.0.3",
  "latestVersionCode": 5,
  "minimumSupportedVersionCode": 1,
  "forceUpdate": false,
  "downloadUrl": "https://lumina-ai.co.in/download-apk?v=5",
  "updatePageUrl": "https://lumina-ai.co.in/update",
  "releaseNotes": []
}
```

The installed app compares `latestVersionCode` with its built-in `currentVersionCode`. If the server version is higher, the app shows an update modal. The update button opens the APK download URL.

Important Android limitation: sideloaded APKs cannot silently update themselves. Android always requires user confirmation before installing an APK. The app can open the official download/install flow, but it cannot reinstall itself in the background.

Current app version:

```text
versionName: 2.0.3
versionCode: 5
```

To prevent repeated update prompts, the backend and deployed secrets must agree with the APK version. Use:

```text
PUBLIC_APP_VERSION_NAME=2.0.3
PUBLIC_APP_VERSION_CODE=5
```

Do not keep older `APP_VERSION_NAME` or `APP_VERSION_CODE` environment variables in Hugging Face secrets, because they can cause stale version metadata.

## Cloudflare Worker

The Worker lives in:

```text
ai_backend/cloudflare-worker/
```

Main files:

```text
wrangler.toml                  Production Worker config and routes
wrangler.workers-dev.toml      workers.dev fallback config
src/worker.js                  Reverse proxy implementation
```

Production routes:

```text
lumina-ai.co.in/*
www.lumina-ai.co.in/*
```

Worker responsibilities:

- Proxy GET, POST, and multipart upload requests to Hugging Face.
- Preserve important headers such as `Authorization`, `Content-Type`, and `Origin`.
- Add CORS headers for allowed origins.
- Enforce a maximum upload size using `MAX_UPLOAD_BYTES`.
- Apply request timeout handling using `REQUEST_TIMEOUT_MS`.
- Return Hugging Face responses transparently.
- Rewrite Hugging Face `Location` redirects to the public Lumina domain.
- Add security headers such as `X-Content-Type-Options` and `Referrer-Policy`.
- Disable caching for `/download-apk` so users receive the newest APK.

Current Worker variables:

```text
ORIGIN_BASE_URL=https://kpatel1607-lumina.hf.space
PUBLIC_BASE_URL=https://lumina-ai.co.in
ALLOWED_ORIGINS=https://lumina-ai.co.in,https://www.lumina-ai.co.in
MAX_UPLOAD_BYTES=15728640
REQUEST_TIMEOUT_MS=120000
```

Deploy the Worker from `ai_backend/cloudflare-worker`:

```bash
wrangler deploy
```

## Hugging Face Deployment

The backend is configured for Hugging Face Spaces using the Docker SDK.

The Dockerfile:

- Uses `python:3.11-slim`.
- Installs `tesseract-ocr`, `libgl1`, and `libglib2.0-0`.
- Installs Python dependencies from `requirements.txt`.
- Runs Uvicorn on port `7860`.
- Keeps heavyweight OCR/layout models disabled by default for memory stability.

Start command:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

Required Hugging Face secrets:

```text
FIREBASE_SERVICE_ACCOUNT_JSON
LUMINA_API_KEY
LUMINA_GENERATION_PROVIDER=gemini
LUMINA_MODEL_NAME=gemini-2.0-flash
PUBLIC_BASE_URL=https://lumina-ai.co.in
CUSTOM_DOMAIN=https://lumina-ai.co.in
ALLOWED_ORIGINS=https://lumina-ai.co.in,https://www.lumina-ai.co.in
ALLOWED_HOSTS=lumina-ai.co.in,www.lumina-ai.co.in,*.hf.space,localhost,127.0.0.1
PUBLIC_APP_VERSION_NAME=2.0.3
PUBLIC_APP_VERSION_CODE=5
```

Optional deployment variables:

```text
APP_DOWNLOAD_PATH=/download-apk
MAX_UPLOAD_BYTES=15728640
LUMINA_ENABLE_STRUCTURE_MODEL=false
LUMINA_ENABLE_PADDLEOCR=false
```

## Backend API

### Health

```text
GET /health
```

Returns backend status, app name, app version, generation provider, and whether the model system is enabled.

### App Version

```text
GET /app-version
```

Returns Android update metadata consumed by the Flutter app.

### Download APK

```text
GET /download-apk
```

Returns `static/Lumina-AI.apk` with APK content type and no-cache headers.

### Generate From Text

```text
POST /v2/generate
Authorization: Bearer <Firebase ID token>
Content-Type: application/json
```

Request body:

```json
{
  "text": "User notes or extracted document text",
  "mode": "student",
  "task": "important_notes"
}
```

Supported modes:

```text
student
professional
general
```

Supported student tasks:

```text
important_notes
qa_generation
answer_questions
flashcards
mcqs
beginner_explanation
revision_sheet
```

Supported professional tasks:

```text
executive_summary
main_points
action_items
meeting_minutes
structured_report
table_format
email_draft
```

Supported general tasks:

```text
short_summary
bullet_summary
key_points
simplify
clean_text
```

The response includes generated markdown/plain text, mode, task, provider, model, sections, usage count, and daily limit details.

### Generate From File

```text
POST /v2/generate-file
Authorization: Bearer <Firebase ID token>
Content-Type: multipart/form-data
```

Accepts uploaded documents/images, extracts text through the backend pipeline, and returns generated output. This endpoint exists for backend-side file extraction, while the current Flutter home screen mostly extracts PDF/image text on-device before sending text to `/v2/generate`.

### Legacy Summary

```text
POST /summarize
```

Legacy summarization route kept for compatibility. New app flows should prefer `/v2/generate`.

## Model System

The model pipeline is inside:

```text
ai_backend/model_systems/
```

Pipeline order:

```text
InputUnderstandingSystem
  -> OCRPipeline
  -> TextCleanupPipeline
  -> SmartTextNormalizer
  -> DocumentStructureParser
  -> SemanticChunker
  -> ModeRouter
  -> PromptBuilder
  -> GenerationService
  -> ResponsePostprocessor
  -> OutputFormatter
```

### Input Understanding

`input_understanding.py` detects input type and chooses extraction strategy. It distinguishes plain text, normal PDFs, scanned PDFs, and images.

### OCR Pipeline

`ocr_pipeline.py` handles extraction:

- Plain text passthrough.
- PDF text extraction with PyMuPDF.
- PDF table detection through PyMuPDF table APIs when available.
- Scanned PDF extraction by rendering pages and applying OCR.
- Image OCR through RapidOCR ONNX by default.
- Optional PaddleOCR path when enabled.
- Tesseract fallback support through the Docker image.

Default deploy-friendly OCR stack:

```text
rapidocr-onnxruntime
pytesseract fallback
PyMuPDF for PDF text/tables
OpenCV/Pillow preprocessing
```

PaddleOCR and PPStructure are intentionally disabled by default because they can exceed memory limits on free or small Hugging Face/Render instances.

### Cleanup and Normalization

`text_cleanup_pipeline.py` and `smart_text_normalizer.py` remove repeated OCR noise, repair common OCR merges, normalize spacing, and prepare cleaner text for structure parsing and generation.

### Document Structure Parser

`document_structure_parser.py` detects document structure such as:

- Title
- Sections and headings
- Paragraphs
- Bullet points
- Numbered lists
- Roman numeral lists
- Questions
- Key-value fields
- Tables inferred from text
- Optional layout blocks from PPStructure

When `LUMINA_ENABLE_STRUCTURE_MODEL=true` and the dependencies are installed, PPStructure can be used for heavier layout analysis. In the current deploy profile it remains off for stability.

### Semantic Chunking

`semantic_chunker.py` converts structured text into chunks such as sections, paragraphs, tables, key-value fields, and layout sections. For long documents it can recommend hierarchical summarization instead of a single generation call.

### Mode Routing

`mode_router.py` maps the user's selected mode and task into a generation target. Student, professional, and general modes use different expectations for tone, structure, and output type.

### Prompt Builder

`prompt_builder.py` creates compact prompts for short/simple text and structured prompts for longer, table-heavy, or layout-rich documents.

Current behavior:

- Text below about 650 words can use compact prompts.
- Long documents and structured documents use semantic chunks and structure summaries.
- Student mode prioritizes study notes, revision, Q&A, MCQs, and flashcards.
- Professional mode prioritizes traceable business output, owners, deadlines, risks, reports, meeting notes, and tables.
- General mode prioritizes concise summaries, simplified text, key points, and cleanup.

### Generation Service

`generation_service.py` sends prompts to the configured model provider.

Current provider path:

```text
LUMINA_GENERATION_PROVIDER=gemini
LUMINA_MODEL_NAME=gemini-2.0-flash
LUMINA_API_KEY=<Gemini API key>
```

An Ollama path exists for local model experiments, with Gemini fallback behavior for cases where small local models are weak on structured prompts.

### Postprocessing and Formatting

`response_postprocessor.py` removes generic AI openings and cleans repeated phrasing.

`output_formatter.py` formats the final output and contains special handling for table output. If a valid markdown table is missing, it tries to build one from detected tables or key-value fields.

This table formatter is one of the key places to improve next, because accurate table reconstruction depends heavily on extraction quality, row/column detection, and stronger layout analysis.

## Flutter App

The app lives in:

```text
notes_summarizer_app/
```

Important files:

```text
lib/main.dart                         Firebase init, theme init, app root
lib/screens/home_screen.dart          Main input/generation workspace
lib/screens/history_screen.dart       Saved summaries, folders, favorites
lib/screens/profile_screen.dart       Profile, theme, legal links, update, account controls
lib/screens/analytics_screen.dart     User summary/folder/mode analytics
lib/screens/about_screen.dart         App information and feedback
lib/screens/onboarding_screen.dart    First-run tour
lib/services/api_service.dart         Backend generation client
lib/services/app_update_service.dart  App version/update client
lib/services/auth_service.dart        Firebase auth/profile/account deletion
lib/services/firebase_service.dart    Firestore summaries/folders/usage
lib/theme/app_theme.dart              Light/dark/appearance theme definitions
lib/theme/theme_controller.dart       Saved theme mode and appearance
```

### App Features

- Email/password signup and login.
- Google sign-in.
- Email verification and password reset.
- Profile update with display name, photo URL, recovery email, and password update.
- Account deletion that removes user profile, summaries, and folders.
- Onboarding/tour for new users.
- Paste text manually.
- Extract PDF text on-device.
- Extract image text with Google ML Kit.
- Capture camera images and extract text.
- Generate AI output in Student, Professional, and General modes.
- Save generated summaries to Firestore.
- View, search, favorite, delete, and open saved summaries.
- Create folders and assign summaries to folders.
- Analytics view for generated documents, folders, favorites, and mode usage.
- About screen with feedback submission.
- Theme mode selection and appearance presets.
- Update prompt through `/app-version`.
- Legal links that open the hosted privacy policy and terms pages.

### Current App API Configuration

`AppUpdateService` uses the public domain:

```text
https://lumina-ai.co.in
```

`ApiService` currently points generation requests directly to:

```text
https://kpatel1607-lumina.hf.space
```

Recommended next cleanup: change generation requests to use the public domain too, so the Flutter app only talks to `https://lumina-ai.co.in` and never exposes the Hugging Face origin.

## Firebase Data Model

The app uses Firebase Auth and Cloud Firestore.

Known collections:

```text
users       User profile data
summaries   Generated documents and saved summaries
folders     User-created folders
usage       Daily usage counters
feedback    Feedback from the About screen
```

Typical `summaries` fields:

```text
uid
userEmail
userName
title
input
summary
markdown
plainText
sections
sectionCount
mode
task
format
provider
model
usageCount
dailyLimit
folder
favorite
type
createdAt
timestamp
```

Typical `users` fields:

```text
uid
name
username
email
photoUrl
provider
emailVerified
recoveryEmail
createdAt
lastLogin
updatedAt
```

## Security Notes

Current security measures:

- Firebase ID tokens are required for protected generation APIs.
- Backend validates modes, tasks, and formats.
- Backend applies request rate limiting.
- Backend tracks per-user daily usage limits.
- Cloudflare Worker limits upload size before proxying large requests.
- Cloudflare Worker preserves auth headers and adds CORS only for allowed origins.
- Cloudflare Worker adds browser security headers.
- APK downloads are served with no-cache headers to avoid stale update loops.
- Secrets are expected to be stored in Hugging Face/Cloudflare/Firebase, not committed.
- Firebase App Check is initialized in the Flutter app.

Sensitive local files exist in the workspace root, including Firebase service-account JSON files and OAuth client secrets. These should not be committed, uploaded publicly, or included inside client apps.

Recommended next security improvements:

- Move the Flutter app into the same repository or a separate private repo with proper `.gitignore` rules.
- Ensure all Firebase service-account files are removed from local shareable folders and rotated if they were ever exposed.
- Use only the Cloudflare public domain in the app.
- Add Firestore security rules review for `users`, `summaries`, `folders`, `usage`, and `feedback`.
- Add Firebase App Check enforcement on Firebase services.
- Add server-side file type validation and stricter MIME sniffing for uploads.
- Add structured logging without storing raw user document text.

## Local Development

### Backend

From `ai_backend`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 7860
```

Useful checks:

```bash
python -m py_compile main.py
node --check cloudflare-worker/src/worker.js
```

### Flutter App

From `notes_summarizer_app`:

```bash
flutter pub get
flutter analyze
flutter run
```

Build Android APK:

```bash
flutter build apk --release --target-platform android-arm64 --no-tree-shake-icons
```

After building a new APK, copy it into:

```text
ai_backend/static/Lumina-AI.apk
```

Then update these version values together:

```text
notes_summarizer_app/pubspec.yaml
notes_summarizer_app/lib/services/app_update_service.dart
ai_backend/main.py default PUBLIC_APP_VERSION_NAME/PUBLIC_APP_VERSION_CODE
Hugging Face PUBLIC_APP_VERSION_NAME/PUBLIC_APP_VERSION_CODE secrets, if used
```

## Deployment Checklist

1. Build the Flutter release APK.
2. Confirm APK metadata versionName/versionCode.
3. Copy the APK to `ai_backend/static/Lumina-AI.apk`.
4. Update backend app version defaults or Hugging Face `PUBLIC_APP_VERSION_*` secrets.
5. Commit backend changes.
6. Push to GitHub.
7. Let Hugging Face rebuild the Docker Space.
8. Deploy the Cloudflare Worker if Worker code changed.
9. Check:

```text
https://lumina-ai.co.in/health
https://lumina-ai.co.in/app-version
https://lumina-ai.co.in/download-app
https://lumina-ai.co.in/privacy-policy
https://lumina-ai.co.in/terms-and-conditions
```

10. Install the downloaded APK on Android and confirm the update modal does not repeat when version codes match.

## Known Gaps And Next Improvements

The next major update should focus on model accuracy, extraction quality, and mode-specific output reliability.

Highest-priority model improvements:

- Replace simple formatting fallbacks with stronger document structure and table reconstruction logic.
- Improve table extraction for professional `table_format`, especially row/column alignment and empty-cell handling.
- Add a dedicated document layout model for PDFs/images when deployment memory allows it.
- Improve OCR quality for scanned PDFs, handwritten notes, low-resolution images, and multi-column documents.
- Add confidence scoring for OCR and extraction quality.
- Add tests using real sample PDFs/images for every mode and task.
- Improve prompt evaluation with expected outputs for student/professional/general tasks.
- Add automatic retries or repair prompts when output violates task format.
- Make `/v2/generate-file` the preferred path for file uploads when backend OCR is stronger than device OCR.

Highest-priority app improvements:

- Add direct file upload to backend `/v2/generate-file` for stronger OCR/document analysis.
- Improve profile photo upload by using Firebase Storage or another secure media store instead of only photo URLs.
- Add richer onboarding steps for modes, folders, history, exports, and updates.
- Add export options such as PDF, TXT, Markdown, and share.
- Add better offline/error states for Hugging Face cold starts.

Highest-priority deployment improvements:

- Keep Hugging Face deployment light by default and enable heavier models only on larger Spaces.
- Consider a GPU Space or separate OCR worker for advanced layout/OCR models.
- Add CI checks for Python compile, Worker syntax, Flutter analyze, and APK version consistency.
- Use Git LFS or release assets for APK distribution if APK size keeps growing.

## Current Version Snapshot

```text
Backend app version: 2.0.3
Android app version: 2.0.3+5
Public domain: https://lumina-ai.co.in
Backend origin: https://kpatel1607-lumina.hf.space
Generation provider: Gemini
Default model: gemini-2.0-flash
Deploy target: Hugging Face Docker Space
Edge proxy: Cloudflare Worker
```
