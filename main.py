from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import os
import re
import html
import json
import tempfile
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from datetime import datetime, timezone
from typing import Dict, Any
from urllib.parse import urlparse

from model_systems.pipeline_router import PipelineRouter


load_dotenv()

APP_NAME = "Lumina AI"
CONTACT_EMAIL = "support@lumina-ai.co.in"
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    os.getenv(
        "BASE_URL",
        "https://lumina-ai.co.in",
    ),
).rstrip("/")
BASE_URL = PUBLIC_BASE_URL

CUSTOM_DOMAIN = os.getenv(
    "CUSTOM_DOMAIN",
    "https://lumina-ai.co.in",
).strip().rstrip("/")
APP_VERSION_NAME = os.getenv("PUBLIC_APP_VERSION_NAME", "2.0.4")
APP_VERSION_CODE = int(os.getenv("PUBLIC_APP_VERSION_CODE", "6"))
APP_DOWNLOAD_PATH = os.getenv("APP_DOWNLOAD_PATH", "/download-apk")
APK_FILE_PATH = os.getenv("APK_FILE_PATH", "static/Lumina-AI.apk")
APP_RELEASE_NOTES = [
    "Improved dark and light mode readability for AI Workspace and output option chips.",
    "Added theme-aware text, icons, borders, and backgrounds across the main workspace.",
    "Added animated depth, pressed-card feedback, and a livelier workspace hero.",
    "Improved card contrast and visual polish across document input, quick actions, usage, folders, and results.",
    "Kept table/output formatting improvements from the previous release.",
]

LUMINA_GENERATION_PROVIDER = os.getenv(
    "LUMINA_GENERATION_PROVIDER",
    "gemini",
).lower().strip()

FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_JSON"
)

if FIREBASE_SERVICE_ACCOUNT_JSON:
    firebase_credentials = credentials.Certificate(
        json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
    )

    if not firebase_admin._apps:
        firebase_admin.initialize_app(firebase_credentials)
else:
    print("WARNING: FIREBASE_SERVICE_ACCOUNT_JSON missing")


ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://ai-notes-summarizer-ck5l.onrender.com,http://localhost:3000,http://localhost:5173,http://localhost:8080",
).split(",")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in ALLOWED_ORIGINS
    if origin.strip()
]

SPACE_HOST = os.getenv("SPACE_HOST", "").strip()

if SPACE_HOST:
    ALLOWED_ORIGINS.append(f"https://{SPACE_HOST}")

if PUBLIC_BASE_URL:
    ALLOWED_ORIGINS.append(PUBLIC_BASE_URL)

if CUSTOM_DOMAIN:
    ALLOWED_ORIGINS.append(CUSTOM_DOMAIN)

ALLOWED_ORIGINS = list(dict.fromkeys(ALLOWED_ORIGINS))

ALLOWED_HOSTS = os.getenv(
    "ALLOWED_HOSTS",
    "localhost,127.0.0.1,*.hf.space,kpatel1607-lumina.hf.space,www.lumina-ai.co.in,lumina-ai.co.in",
).split(",")

ALLOWED_HOSTS = [
    host.strip()
    for host in ALLOWED_HOSTS
    if host.strip()
]

for configured_url in [PUBLIC_BASE_URL, CUSTOM_DOMAIN]:
    parsed_host = urlparse(configured_url).netloc
    if parsed_host and parsed_host not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append(parsed_host)

MAX_INPUT_LENGTH = int(
    os.getenv("MAX_INPUT_LENGTH", "45000")
)

MAX_UPLOAD_BYTES = int(
    os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))
)

DAILY_FREE_LIMIT = int(
    os.getenv("DAILY_FREE_LIMIT", "15")
)

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
)

app = FastAPI(
    title="Lumina AI API",
    description="AI-powered OCR cleanup and academic summarization backend for Lumina.",
    version="2.0.0",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=ALLOWED_HOSTS,
)
if os.path.isdir("static"):
    app.mount(
        "/static",
        StaticFiles(directory="static"),
        name="static",
    )


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )

    return response


def public_base_url() -> str:
    return PUBLIC_BASE_URL


def app_download_url() -> str:
    if APP_DOWNLOAD_PATH.startswith("http"):
        return APP_DOWNLOAD_PATH

    return f"{public_base_url()}{APP_DOWNLOAD_PATH}?v={APP_VERSION_CODE}"


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(
    request: Request,
    exc: RateLimitExceeded,
):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Too many requests. Please try again shortly.",
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


lumina_router = PipelineRouter()


class NoteRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=MAX_INPUT_LENGTH,
    )

    format: str = Field(
        default="bullet",
        max_length=30,
    )


class GenerateRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=MAX_INPUT_LENGTH,
    )

    mode: str = Field(
        default="student",
        max_length=30,
    )

    task: str = Field(
        default="important_notes",
        max_length=50,
    )


def require_firebase() -> None:
    if firebase_admin._apps and firestore_db is not None:
        return

    raise HTTPException(
        status_code=503,
        detail=(
            "Authentication service is not configured. "
            "Please set FIREBASE_SERVICE_ACCOUNT_JSON."
        ),
    )


def verify_firebase_user(
    authorization: str = Header(None),
):
    require_firebase()

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Please login first",
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format",
        )

    id_token = authorization.replace(
        "Bearer ",
        "",
        1,
    ).strip()

    try:
        decoded_token = firebase_auth.verify_id_token(
            id_token
        )

        return decoded_token

    except Exception as e:
        print("Firebase token verification error:", str(e))

        raise HTTPException(
            status_code=401,
            detail=str(e),
        )


firestore_db = None

if firebase_admin._apps:
    firestore_db = firestore.client()


def clean_input_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"([^\w\s])\1{4,}", r"\1", text)
    text = text.strip()

    return text


def clean_ai_output(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    unwanted_starts = [
        "Here is the summary:",
        "Here's the summary:",
        "Here are the notes:",
        "Here is the cleaned version:",
        "Sure,",
        "Sure.",
        "Of course,",
        "The summary is:",
    ]

    for phrase in unwanted_starts:
        if text.lower().startswith(phrase.lower()):
            text = text[len(phrase):].strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.replace("• •", "•")
    text = text.strip()

    return text


def extract_generation_error(result: Dict[str, Any]) -> str:
    generation_result = result.get("generation_result", {})

    if isinstance(generation_result, dict):
        error = generation_result.get("error")

        if error:
            return str(error)

    errors = result.get("errors")

    if errors:
        return str(errors)

    return "AI returned empty output"


def validate_format(format_value: str) -> str:
    allowed = {
        "bullet",
        "short",
        "detailed",
        "keypoints",
        "beginner",
        "qa",
    }

    cleaned = format_value.lower().strip()

    if cleaned not in allowed:
        return "bullet"

    return cleaned


def validate_mode(mode: str) -> str:
    allowed = {
        "student",
        "professional",
        "general",
    }

    cleaned = mode.lower().strip()

    if cleaned not in allowed:
        return "student"

    return cleaned


def validate_task(mode: str, task: str) -> str:
    cleaned_task = task.lower().strip()

    allowed_tasks = {
        "student": {
            "important_notes",
            "qa_generation",
            "answer_questions",
            "flashcards",
            "mcqs",
            "beginner_explanation",
            "revision_sheet",
        },
        "professional": {
            "executive_summary",
            "main_points",
            "action_items",
            "meeting_minutes",
            "structured_report",
            "table_format",
            "email_draft",
        },
        "general": {
            "short_summary",
            "bullet_summary",
            "key_points",
            "simplify",
            "clean_text",
        },
    }

    defaults = {
        "student": "important_notes",
        "professional": "executive_summary",
        "general": "short_summary",
    }

    if cleaned_task not in allowed_tasks.get(mode, set()):
        return defaults.get(mode, "short_summary")

    return cleaned_task


def site_styles() -> str:
    return """
    <style>
        * { box-sizing: border-box; }
        :root {
            --ink: #101828;
            --muted: #667085;
            --line: rgba(16, 24, 40, .12);
            --panel: rgba(255, 255, 255, .78);
            --brand: #315efb;
            --mint: #00a88f;
            --gold: #f59e0b;
            --rose: #e11d48;
        }
        html { scroll-behavior: smooth; }
        body {
            margin: 0;
            min-height: 100vh;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 10%, rgba(49, 94, 251, .14), transparent 28%),
                radial-gradient(circle at 86% 12%, rgba(0, 168, 143, .13), transparent 30%),
                linear-gradient(135deg, #f8fbff 0%, #ffffff 48%, #f5faf8 100%);
            overflow-x: hidden;
        }
        body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            opacity: .34;
            background-image:
                linear-gradient(rgba(16,24,40,.045) 1px, transparent 1px),
                linear-gradient(90deg, rgba(16,24,40,.045) 1px, transparent 1px);
            background-size: 42px 42px;
            mask-image: linear-gradient(to bottom, black, transparent 78%);
        }
        a { color: inherit; text-decoration: none; }
        .nav {
            position: sticky;
            top: 0;
            z-index: 20;
            backdrop-filter: blur(18px);
            background: rgba(255,255,255,.74);
            border-bottom: 1px solid var(--line);
        }
        .nav-inner, .wrap {
            width: min(1120px, calc(100% - 36px));
            margin: 0 auto;
        }
        .nav-inner {
            min-height: 72px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 18px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 900;
            letter-spacing: 0;
        }
        .logo {
            width: 38px;
            height: 38px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            color: white;
            background: conic-gradient(from 190deg, var(--brand), var(--mint), var(--gold), var(--brand));
            box-shadow: 0 14px 34px rgba(49,94,251,.22);
            animation: floaty 4s ease-in-out infinite;
        }
        .links {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .links a, .pill {
            border: 1px solid var(--line);
            background: rgba(255,255,255,.72);
            padding: 10px 14px;
            border-radius: 999px;
            color: #344054;
            font-weight: 700;
            font-size: 14px;
        }
        .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            border: 0;
            border-radius: 999px;
            background: #101828;
            color: white;
            padding: 14px 19px;
            font-weight: 900;
            box-shadow: 0 18px 36px rgba(16,24,40,.18);
            cursor: pointer;
            transition: transform .2s ease, box-shadow .2s ease;
        }
        .button:hover { transform: translateY(-2px); box-shadow: 0 22px 42px rgba(16,24,40,.24); }
        .button.secondary {
            background: rgba(255,255,255,.78);
            color: var(--ink);
            border: 1px solid var(--line);
            box-shadow: none;
        }
        .hero {
            min-height: calc(100vh - 72px);
            display: grid;
            grid-template-columns: minmax(0, 1.03fr) minmax(320px, .97fr);
            gap: 38px;
            align-items: center;
            padding: 58px 0 42px;
        }
        .eyebrow {
            display: inline-flex;
            gap: 8px;
            align-items: center;
            color: #175cd3;
            background: rgba(49,94,251,.09);
            border: 1px solid rgba(49,94,251,.16);
            padding: 9px 13px;
            border-radius: 999px;
            font-weight: 850;
            font-size: 14px;
        }
        h1 {
            margin: 18px 0 16px;
            font-size: clamp(44px, 8vw, 82px);
            line-height: .92;
            letter-spacing: 0;
        }
        .lead {
            max-width: 680px;
            color: var(--muted);
            font-size: clamp(17px, 2vw, 20px);
            line-height: 1.7;
        }
        .hero-actions {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 26px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin-top: 28px;
        }
        .metric, .card, .device, .legal-card {
            background: var(--panel);
            border: 1px solid var(--line);
            box-shadow: 0 20px 52px rgba(16,24,40,.08);
            backdrop-filter: blur(18px);
        }
        .metric {
            border-radius: 18px;
            padding: 15px;
        }
        .metric strong { display: block; font-size: 22px; }
        .metric span { color: var(--muted); font-size: 13px; }
        .device {
            position: relative;
            border-radius: 34px;
            padding: 18px;
            overflow: hidden;
            animation: riseIn .7s ease both;
        }
        .device::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(115deg, transparent 0%, rgba(255,255,255,.7) 48%, transparent 56%);
            transform: translateX(-100%);
            animation: sheen 5s ease-in-out infinite;
            pointer-events: none;
        }
        .screen {
            border-radius: 24px;
            background: #0f172a;
            color: white;
            padding: 20px;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }
        .scan-card {
            background: rgba(255,255,255,.08);
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 20px;
            padding: 16px;
            position: relative;
            overflow: hidden;
        }
        .scan-line {
            position: absolute;
            left: 12px;
            right: 12px;
            height: 2px;
            background: #34d399;
            box-shadow: 0 0 26px #34d399;
            animation: scan 3.2s ease-in-out infinite;
        }
        .fake-line {
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,.22);
            margin: 11px 0;
        }
        .fake-line.short { width: 62%; }
        .mode-tabs {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }
        .mode-tabs button {
            border: 1px solid rgba(255,255,255,.14);
            background: rgba(255,255,255,.08);
            color: white;
            border-radius: 14px;
            padding: 11px 8px;
            font-weight: 800;
            cursor: pointer;
        }
        .mode-tabs button.active {
            background: #ffffff;
            color: #101828;
        }
        .output {
            flex: 1;
            background: white;
            color: #101828;
            border-radius: 20px;
            padding: 18px;
        }
        .output h3 { margin: 0 0 10px; }
        .output ul { margin: 0; padding-left: 18px; color: #344054; line-height: 1.7; }
        .section {
            padding: 46px 0;
        }
        .section h2 {
            margin: 0 0 12px;
            font-size: clamp(28px, 4vw, 46px);
        }
        .section-lead {
            color: var(--muted);
            max-width: 760px;
            line-height: 1.7;
            font-size: 17px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 24px;
        }
        .card, .legal-card {
            border-radius: 22px;
            padding: 22px;
            transition: transform .2s ease, border-color .2s ease;
        }
        .card:hover, .legal-card:hover {
            transform: translateY(-4px);
            border-color: rgba(49,94,251,.32);
        }
        .icon {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: rgba(49,94,251,.1);
            margin-bottom: 14px;
        }
        .card p, .legal-card p { color: var(--muted); line-height: 1.65; }
        .status-row {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-top: 18px;
        }
        .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #f59e0b;
            box-shadow: 0 0 0 6px rgba(245,158,11,.14);
        }
        .dot.good {
            background: #12b76a;
            box-shadow: 0 0 0 6px rgba(18,183,106,.14);
        }
        .footer {
            border-top: 1px solid var(--line);
            padding: 26px 0 40px;
            color: var(--muted);
        }
        .legal-layout {
            padding: 38px 0 56px;
        }
        .legal-card {
            width: min(940px, calc(100% - 36px));
            margin: 0 auto;
        }
        .legal-card h1 {
            font-size: clamp(34px, 6vw, 58px);
            line-height: 1;
        }
        .legal-card h2 { margin-top: 30px; }
        .legal-card li { margin-bottom: 10px; line-height: 1.75; color: var(--muted); }
        .notice {
            border: 1px solid rgba(49,94,251,.18);
            background: rgba(49,94,251,.08);
            border-radius: 16px;
            padding: 16px;
            color: #344054;
        }
        @keyframes floaty {
            0%, 100% { transform: translateY(0) rotate(0); }
            50% { transform: translateY(-4px) rotate(3deg); }
        }
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(18px) scale(.98); }
            to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes sheen {
            0%, 55% { transform: translateX(-110%); }
            82%, 100% { transform: translateX(110%); }
        }
        @keyframes scan {
            0%, 100% { top: 18px; opacity: .4; }
            50% { top: calc(100% - 20px); opacity: 1; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation: none !important; transition: none !important; }
        }
        @media (max-width: 860px) {
            .hero { grid-template-columns: 1fr; min-height: auto; }
            .grid, .metrics { grid-template-columns: 1fr; }
            .screen { min-height: 420px; }
        }
    </style>
"""


def site_nav(back_href: str | None = None) -> str:
    if back_href:
        nav_action = f'<a href="{html.escape(back_href)}">Back</a>'
    else:
        nav_action = '<a href="/download-app">Download</a>'

    return """
    <header class="nav">
        <div class="nav-inner">
            <a class="brand" href="/">
                <span class="logo">L</span>
                <span>Lumina AI</span>
            </a>
            <div class="links">
                __NAV_ACTION__
            </div>
        </div>
    </header>
""".replace("__NAV_ACTION__", nav_action)


def site_scripts() -> str:
    return """
    <script>
        const previews = {
            student: {
                title: "Student Mode",
                points: ["Exam-ready notes", "Flashcards and Q&A", "Beginner explanations"]
            },
            professional: {
                title: "Professional Mode",
                points: ["Meeting minutes", "Action items", "Structured reports and tables"]
            },
            general: {
                title: "General Mode",
                points: ["Short summaries", "Cleaned text", "Key points in seconds"]
            }
        };

        function setMode(mode) {
            document.querySelectorAll("[data-mode]").forEach((button) => {
                button.classList.toggle("active", button.dataset.mode === mode);
            });

            const preview = previews[mode];
            const output = document.getElementById("mode-output");
            if (!output) return;

            output.innerHTML = `<h3>${preview.title}</h3><ul>${preview.points.map((point) => `<li>${point}</li>`).join("")}</ul>`;
        }

        async function loadStatus() {
            const label = document.getElementById("status-label");
            const dot = document.getElementById("status-dot");
            const version = document.getElementById("version-label");

            try {
                const [health, appVersion] = await Promise.all([
                    fetch("/health").then((response) => response.json()),
                    fetch("/app-version").then((response) => response.json())
                ]);

                if (health.status === "healthy") {
                    label.textContent = "Cloudflare proxy and AI backend are online";
                    dot.classList.add("good");
                }

                version.textContent = `Android v${appVersion.latestVersionName || "2.0.0"}`;
            } catch (error) {
                label.textContent = "Status check is temporarily unavailable";
            }
        }

        setMode("student");
        loadStatus();
    </script>
"""


def legal_page(
    title: str,
    subtitle: str,
    body: str,
) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{html.escape(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {site_styles()}
</head>
<body>
    {site_nav("/")}
    <main class="legal-layout">
      <article class="legal-card">
        <span class="eyebrow">Lumina AI legal</span>
        <h1>{html.escape(title)}</h1>
        <p class="lead">{html.escape(subtitle)}</p>
        {body}
        <div class="footer">
            © 2026 Lumina AI. All rights reserved.
            <br>
            Contact: <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </div>
      </article>
    </main>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Lumina AI - AI Notes Summarizer</title>
    <meta name="description" content="Lumina AI converts notes, PDFs, scanned pages, and images into clean AI-powered summaries for study and productivity.">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {site_styles()}
</head>
<body>
    {site_nav()}
    <main class="wrap">
        <section class="hero">
            <div>
                <span class="eyebrow">AI study workspace for Android</span>
                <h1>Lumina AI</h1>
                <p class="lead">
                    Turn notes, PDFs, camera scans, and images into clean summaries,
                    flashcards, Q&A, revision sheets, tables, and professional reports.
                    Built for students and busy teams who need organized output fast.
                </p>
                <div class="hero-actions">
                    <a class="button" href="/download-app">Download app</a>
                    <a class="button secondary" href="/privacy-policy">View privacy policy</a>
                </div>
                <div class="status-row">
                    <span id="status-dot" class="dot"></span>
                    <span id="status-label">Checking Cloudflare proxy and backend...</span>
                    <span class="pill" id="version-label">Android v{APP_VERSION_NAME}</span>
                </div>
                <div class="metrics">
                    <div class="metric"><strong>3</strong><span>AI modes</span></div>
                    <div class="metric"><strong>OCR</strong><span>PDF and image extraction</span></div>
                    <div class="metric"><strong>Cloud</strong><span>Folders, favorites, history</span></div>
                </div>
            </div>
            <div class="device" aria-label="Lumina AI app preview">
                <div class="screen">
                    <div class="scan-card">
                        <div class="scan-line"></div>
                        <strong>Document scan</strong>
                        <div class="fake-line"></div>
                        <div class="fake-line"></div>
                        <div class="fake-line short"></div>
                    </div>
                    <div class="mode-tabs">
                        <button class="active" data-mode="student" onclick="setMode('student')">Student</button>
                        <button data-mode="professional" onclick="setMode('professional')">Pro</button>
                        <button data-mode="general" onclick="setMode('general')">General</button>
                    </div>
                    <div id="mode-output" class="output"></div>
                </div>
            </div>
        </section>
        <section class="section" id="features">
            <h2>Everything organized after generation.</h2>
            <p class="section-lead">
                Lumina AI is more than a generate button. It extracts text,
                understands document structure, formats outputs by mode, and
                keeps generated notes inside a searchable workspace.
            </p>
            <div class="grid">
                <div class="card"><div class="icon">OCR</div><h3>Sharper extraction</h3><p>PDF text, scanned images, table hints, and OCR cleanup work together before generation.</p></div>
                <div class="card"><div class="icon">AI</div><h3>Mode-aware outputs</h3><p>Student, professional, and general tasks use different instructions and formatting rules.</p></div>
                <div class="card"><div class="icon">DIR</div><h3>Saved workspace</h3><p>Folders, favorites, history, analytics, and account controls keep summaries easy to find.</p></div>
            </div>
        </section>
        <section class="section">
            <h2>Built for app store trust.</h2>
            <p class="section-lead">
                Public privacy, terms, update, and download pages are served from
                your own domain through Cloudflare while the AI backend stays on Hugging Face.
            </p>
            <div class="grid">
                <a class="legal-card" href="/privacy-policy"><h3>Privacy Policy</h3><p>Data collection, Firebase authentication, summaries, folders, analytics, and deletion rights.</p></a>
                <a class="legal-card" href="/terms-and-conditions"><h3>Terms</h3><p>Acceptable use, AI limitations, account responsibility, updates, and service availability.</p></a>
                <a class="legal-card" href="/app-version"><h3>Update API</h3><p>Version metadata used by the Flutter app to send users to the latest release.</p></a>
            </div>
        </section>
    </main>
    <footer class="wrap footer">
        Lumina AI. Contact <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
    </footer>
    {site_scripts()}
</body>
</html>
"""


@app.get("/health")
def health_check():
    return {
        "app": APP_NAME,
        "status": "healthy",
        "version": APP_VERSION_NAME,
        "generation_provider": LUMINA_GENERATION_PROVIDER,
        "model_system_enabled": True,
    }


@app.get("/app-version")
def app_version():
    return {
        "app": APP_NAME,
        "latestVersionName": APP_VERSION_NAME,
        "latestVersionCode": APP_VERSION_CODE,
        "minimumSupportedVersionCode": 1,
        "forceUpdate": False,
        "downloadUrl": app_download_url(),
        "updatePageUrl": f"{BASE_URL}/update",
        "releaseNotes": APP_RELEASE_NOTES,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/download-apk")
def download_apk():
    if not os.path.exists(APK_FILE_PATH):
        raise HTTPException(
            status_code=404,
            detail="APK file is not available on this deployment.",
        )

    return FileResponse(
        APK_FILE_PATH,
        media_type="application/vnd.android.package-archive",
        filename=f"Lumina-AI-v{APP_VERSION_NAME}.apk",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Lumina-App-Version": APP_VERSION_NAME,
            "X-Lumina-App-Version-Code": str(APP_VERSION_CODE),
        },
    )


@app.get("/download-app", response_class=HTMLResponse)
def download_app():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Download Lumina AI</title>
    <meta name="description" content="Download the latest Lumina AI Android app.">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {site_styles()}
</head>
<body>
    {site_nav("/")}
    <main class="wrap">
        <section class="hero">
            <div>
                <span class="eyebrow">Latest Android release</span>
                <h1>Lumina AI</h1>
                <p class="lead">
                    Convert notes, PDFs, camera scans, and images into structured summaries,
                    Q&A, flashcards, revision sheets, and professional reports.
                </p>
                <a class="button" href="{APP_DOWNLOAD_PATH}" download>Download APK v{APP_VERSION_NAME}</a>
                <div class="status-row">
                    <span id="status-dot" class="dot"></span>
                    <span id="status-label">Checking latest release metadata...</span>
                    <span class="pill" id="version-label">Version code {APP_VERSION_CODE}</span>
                </div>
            </div>
            <aside class="device">
                <div class="screen">
                    <h2>Release Notes</h2>
                    <ul>
                        {''.join(f'<li>{html.escape(note)}</li>' for note in APP_RELEASE_NOTES)}
                    </ul>
                    <div class="scan-card">
                        <div class="scan-line"></div>
                        <strong>Update flow</strong>
                        <div class="fake-line"></div>
                        <div class="fake-line short"></div>
                    </div>
                </div>
            </aside>
        </section>
        <section class="grid">
            <div class="card"><div class="icon">ID</div><strong>Private accounts</strong><p>Email/Google sign-in, account deletion, and saved history controls.</p></div>
            <div class="card"><div class="icon">DIR</div><strong>Organized documents</strong><p>Folders, favorites, search, and generated-document history.</p></div>
            <div class="card"><div class="icon">AI</div><strong>AI modes</strong><p>Student, professional, and general outputs tuned for different workflows.</p></div>
        </section>
    </main>
    <footer class="wrap footer">
        <a href="/privacy-policy">Privacy Policy</a> | <a href="/terms-and-conditions">Terms & Conditions</a>
    </footer>
    {site_scripts()}
</body>
</html>
"""


@app.get("/update", response_class=HTMLResponse)
def update_app():
    return download_app()


@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    body = f"""
        <p>
            Lumina AI respects user privacy and collects only the information
            needed to provide authentication, AI summarization, saved document
            history, folders, favorites, analytics, safety, and account support.
        </p>

        <div class="notice">
            This policy is written for the Lumina AI app, backend API, policy
            website, APK download page, and related Firebase-backed services.
        </div>

        <h2>Information We Collect</h2>
        <ul>
            <li>Account information such as name, username, email address, profile photo URL, provider, email verification status, account creation date, and last login time.</li>
            <li>Authentication information handled by Firebase Authentication, including email/password login, Google sign-in, verification email status, password reset flows, and session tokens.</li>
            <li>Notes, pasted text, extracted PDF text, camera OCR text, uploaded image OCR text, selected AI mode, selected output task, generated summaries, markdown, plain text, sections, and word-count metadata.</li>
            <li>Document organization data such as folders, favorites, document history, creation timestamps, and user-specific usage counters.</li>
            <li>App diagnostics and analytics events such as app opens, crashes, performance errors, update prompts, and feature usage where Firebase Analytics or Crashlytics are enabled.</li>
            <li>Technical request information such as IP-derived rate-limit data, request timing, API errors, and service health information needed to protect the backend.</li>
        </ul>

        <h2>How We Use Information</h2>
        <ul>
            <li>To authenticate users and protect account access.</li>
            <li>To extract text from images, camera scans, and PDFs on the device where supported.</li>
            <li>To send notes and extracted text to the backend for AI summarization and formatting.</li>
            <li>To save generated documents, folders, favorites, and usage limits to the user's account.</li>
            <li>To detect crashes, measure reliability, prevent abuse, apply rate limits, and improve the product experience.</li>
            <li>To notify users about important app updates and route them to the official download page.</li>
        </ul>

        <h2>AI Processing</h2>
        <p>
            Text submitted for summarization may be processed by AI model providers
            used by Lumina AI, including Gemini or compatible configured model
            providers. Outputs can be inaccurate or incomplete, so users should
            review important results before relying on them. Users should avoid
            uploading highly sensitive, confidential, illegal, medical, legal,
            financial, or restricted content unless they have the right to do so.
        </p>

        <h2>Third-Party Services</h2>
        <ul>
            <li>Firebase Authentication for sign-in, verification, password reset, and account identity.</li>
            <li>Cloud Firestore for user profiles, summaries, folders, favorites, and daily usage counters.</li>
            <li>Firebase App Check, Analytics, and Crashlytics for abuse protection, diagnostics, app-open analytics, and crash reports.</li>
            <li>Google sign-in where the user chooses Google authentication.</li>
            <li>Google ML Kit or device OCR libraries for image/camera text recognition in the app.</li>
            <li>Configured AI model providers for summarization and output generation.</li>
        </ul>

        <h2>Data Retention</h2>
        <p>
            Saved summaries, folders, favorites, and profile records are retained
            while the account remains active. Users can delete individual documents
            or delete their account from the Profile section. Usage logs and
            diagnostic records may remain for a limited period where required for
            security, fraud prevention, legal compliance, or service reliability.
        </p>

        <h2>Data Security</h2>
        <p>
            Lumina AI uses account-based storage and user identifiers to keep saved
            summaries separated between users. API requests require Firebase
            authentication, the backend applies rate limits, and Firebase App Check
            can be used to reduce unauthorized access. No online system can be
            guaranteed completely secure. We do not sell user data.
        </p>

        <h2>User Control</h2>
        <ul>
            <li>Users may delete generated summaries and folders from inside the app.</li>
            <li>Users may reset passwords, update profile information, and manage sign-in methods supported by Firebase.</li>
            <li>Users may delete their account and associated saved data from the Profile section.</li>
            <li>Users may contact support for privacy questions or deletion assistance.</li>
        </ul>

        <h2>Children's Privacy</h2>
        <p>
            Lumina AI is intended for general study and productivity use. Children
            should use the service only with appropriate parent, guardian, or school
            permission where required by law.
        </p>

        <h2>International Processing</h2>
        <p>
            Data may be processed by cloud providers and AI services in regions
            outside the user's location, subject to those providers' safeguards and
            terms.
        </p>

        <h2>Contact</h2>
        <p>
            For privacy questions, contact:
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </p>
    """

    return legal_page(
        "Privacy Policy",
        "Last updated: 2026",
        body,
    )


@app.get("/terms-and-conditions", response_class=HTMLResponse)
def terms_and_conditions():
    body = f"""
        <p>
            By using Lumina AI, you agree to these Terms & Conditions.
            If you do not agree, please do not use the app.
        </p>

        <h2>Use of Service</h2>
        <p>
            Lumina AI provides tools for OCR-based note extraction, AI-powered
            summarization, study note generation, and productivity support.
        </p>

        <h2>User Responsibilities</h2>
        <ul>
            <li>Users must not upload illegal, harmful, abusive, or infringing content.</li>
            <li>Users are responsible for the content they upload or paste.</li>
            <li>Users must have the right to upload, process, summarize, or store the documents they submit.</li>
            <li>Users must not attempt to bypass rate limits, abuse backend APIs, reverse engineer protected services, or disrupt Lumina AI infrastructure.</li>
            <li>Users should verify important academic, legal, medical, or professional information independently.</li>
        </ul>

        <h2>AI Generated Content</h2>
        <p>
            AI-generated summaries may contain mistakes, omissions, or imperfect
            interpretations. Lumina AI is a study assistant, not a replacement for
            professional, academic, legal, or medical advice.
        </p>

        <h2>Accounts and Security</h2>
        <p>
            Users are responsible for maintaining account security, using accurate
            sign-in information, and protecting devices where Lumina AI is installed.
            Password reset, email verification, and recovery email records depend
            on Firebase and platform availability.
        </p>

        <h2>Account Security</h2>
        <p>
            Users are responsible for maintaining the confidentiality of their login
            credentials and account access.
        </p>

        <h2>Data Deletion</h2>
        <p>
            Users may delete their account and associated saved data from the app's
            Profile section. Some third-party providers may retain limited records
            according to their own policies.
        </p>

        <h2>Downloads and Updates</h2>
        <p>
            Lumina AI may provide update notices through the app and an official
            download page. Users should install updates only from Lumina AI's
            official website or official app-store listing when available. APK
            installation may require Android device permissions controlled by the
            operating system.
        </p>

        <h2>Free Limits and Fair Use</h2>
        <p>
            Lumina AI may apply daily generation limits, request limits, and other
            safeguards to keep the service reliable. Limits may change as the app
            evolves.
        </p>

        <h2>Intellectual Property</h2>
        <p>
            Lumina AI, its branding, app design, backend, and website are owned by
            their respective rights holders. Users retain responsibility for their
            uploaded content and must respect third-party copyrights.
        </p>

        <h2>Disclaimer and Liability</h2>
        <p>
            Lumina AI is provided on an "as available" basis. To the maximum extent
            permitted by law, Lumina AI is not liable for losses arising from
            inaccurate AI outputs, user-submitted content, service interruptions,
            third-party services, or unsupported device configurations.
        </p>

        <h2>Availability</h2>
        <p>
            Lumina AI may occasionally be unavailable due to maintenance, third-party
            service issues, or technical limitations.
        </p>

        <h2>Changes to Terms</h2>
        <p>
            We may update these terms as the app evolves. Continued use of Lumina AI
            means you accept the latest version of these terms.
        </p>

        <h2>Contact</h2>
        <p>
            For questions, contact:
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </p>
    """

    return legal_page(
        "Terms & Conditions",
        "Last updated: 2026",
        body,
    )


summary_styles = {
    "bullet": """
Create clean structured bullet-point study notes.

Output rules:
- Use short, meaningful bullet points
- Preserve all important facts, definitions, formulas, dates, and concepts
- Group related points under clear headings if useful
- Remove repetition and OCR noise
- Keep wording clear and revision-friendly
""",

    "short": """
Create a concise summary.

Output rules:
- Focus only on the core ideas
- Use short paragraphs or compact bullets
- Remove examples unless they are essential
- Preserve the main meaning accurately
- Keep it fast to revise
""",

    "detailed": """
Create detailed study notes.

Output rules:
- Use clear headings and subheadings
- Explain concepts accurately
- Preserve important examples, formulas, facts, and definitions
- Keep logical order from the original text
- Make it suitable for exam preparation
""",

    "keypoints": """
Extract only the most important key points.

Output rules:
- Prioritize facts, formulas, definitions, processes, and comparisons
- Keep points compact and high-value
- Avoid long explanations
- Remove filler and repeated ideas
""",

    "beginner": """
Explain the content for a complete beginner.

Output rules:
- Use simple language
- Explain difficult terms briefly
- Break complex ideas into small steps
- Avoid unnecessary jargon
- Keep it friendly but not childish
""",

    "qa": """
Convert the notes into study question-answer format.

Output rules:
- Generate meaningful questions from the provided text only
- Give concise and accurate answers
- Cover all major concepts
- Avoid inventing information
- Format as Q1/A1, Q2/A2, etc.
""",
}


def check_and_increment_daily_usage(
    user_uid: str,
):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    usage_ref = firestore_db.collection("usage").document(
        f"{user_uid}_{today}"
    )

    transaction = firestore_db.transaction()

    @firestore.transactional
    def update_usage(transaction, usage_ref):
        snapshot = usage_ref.get(
            transaction=transaction,
        )

        if snapshot.exists:
            data = snapshot.to_dict() or {}
            current_count = data.get("count", 0)

            if current_count >= DAILY_FREE_LIMIT:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Daily free limit reached. "
                        f"You can generate {DAILY_FREE_LIMIT} summaries per day."
                    ),
                )

            transaction.update(
                usage_ref,
                {
                    "count": current_count + 1,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )

            return current_count + 1

        transaction.set(
            usage_ref,
            {
                "uid": user_uid,
                "date": today,
                "count": 1,
                "limit": DAILY_FREE_LIMIT,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

        return 1

    return update_usage(
        transaction,
        usage_ref,
    )


@app.post("/v2/generate")
@limiter.limit("20/minute")
def generate_v2(
    request: Request,
    generate_request: GenerateRequest,
    authorization: str = Header(None),
):
    decoded_user = verify_firebase_user(
        authorization,
    )

    user_uid = decoded_user.get("uid")

    if not user_uid:
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase user",
        )

    cleaned_text = clean_input_text(
        generate_request.text,
    )

    if len(cleaned_text) < 5:
        raise HTTPException(
            status_code=400,
            detail="Input text is too short",
        )

    if len(cleaned_text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Input text is too large",
        )

    selected_mode = validate_mode(
        generate_request.mode,
    )

    selected_task = validate_task(
        selected_mode,
        generate_request.task,
    )

    try:
        result = lumina_router.generate_from_text(
            text=cleaned_text,
            mode=selected_mode,
            task=selected_task,
        )

        formatted = result.get(
            "formatted_output",
            {},
        )

        generated_text = formatted.get(
            "markdown",
            "",
        )

        if not generated_text:
            error_detail = extract_generation_error(result)
            print("V2 empty output details:", error_detail)

            raise HTTPException(
                status_code=500,
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
        )

        return JSONResponse(
            content={
                "success": True,
                "title": formatted.get("title", ""),
                "markdown": formatted.get("markdown", ""),
                "plainText": formatted.get("plain_text", ""),
                "sections": formatted.get("sections", []),
                "sectionCount": formatted.get("section_count", 0),
                "mode": formatted.get("mode", selected_mode),
                "task": formatted.get("task", selected_task),
                "format": formatted.get("format", ""),
                "provider": formatted.get("provider", ""),
                "model": formatted.get("model", ""),
                "usageCount": usage_count,
                "dailyLimit": DAILY_FREE_LIMIT,
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print("V2 generation error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Failed to generate output. Please try again.",
        )


@app.post("/v2/generate-file")
@limiter.limit("10/minute")
async def generate_file_v2(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form(default="student"),
    task: str = Form(default="important_notes"),
    authorization: str = Header(None),
):
    decoded_user = verify_firebase_user(
        authorization,
    )

    user_uid = decoded_user.get("uid")

    if not user_uid:
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase user",
        )

    selected_mode = validate_mode(mode)
    selected_task = validate_task(selected_mode, task)

    filename = file.filename or "upload"
    extension = os.path.splitext(filename)[1].lower()

    if extension not in {
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    temp_path = None

    try:
        total_size = 0

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_path = temp_file.name

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file is too large. "
                            f"Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                        ),
                    )

                temp_file.write(chunk)

            if total_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is empty",
                )

        result = lumina_router.generate_from_file(
            file_path=temp_path,
            mode=selected_mode,
            task=selected_task,
        )

        formatted = result.get("formatted_output", {})
        generated_text = formatted.get("markdown", "")

        if not generated_text:
            error_detail = extract_generation_error(result)

            raise HTTPException(
                status_code=500,
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(user_uid)
        pipeline_output = result.get("pipeline_output", {})
        extraction = pipeline_output.get("extraction", {})
        structure = pipeline_output.get("structure", {})

        return JSONResponse(
            content={
                "success": True,
                "title": formatted.get("title", ""),
                "markdown": formatted.get("markdown", ""),
                "plainText": formatted.get("plain_text", ""),
                "sections": formatted.get("sections", []),
                "sectionCount": formatted.get("section_count", 0),
                "mode": formatted.get("mode", selected_mode),
                "task": formatted.get("task", selected_task),
                "format": formatted.get("format", ""),
                "provider": formatted.get("provider", ""),
                "model": formatted.get("model", ""),
                "usageCount": usage_count,
                "dailyLimit": DAILY_FREE_LIMIT,
                "extractionSource": extraction.get("source", ""),
                "extractionConfidence": extraction.get("confidence", 0),
                "tableCount": structure.get("metadata", {}).get(
                    "table_count",
                    0,
                ),
                "parserType": structure.get("parser_type", ""),
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print("V2 file generation error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded file. Please try again.",
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/summarize")
@limiter.limit("20/minute")
def summarize_notes(
    request: Request,
    note_request: NoteRequest,
    authorization: str = Header(None),
):
    """
    Legacy endpoint kept for older app versions.

    It no longer uses legacy AI client.
    It maps old summary formats to the new Lumina model-system pipeline.
    """

    decoded_user = verify_firebase_user(
        authorization,
    )

    user_uid = decoded_user.get("uid")

    if not user_uid:
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase user",
        )

    cleaned_text = clean_input_text(
        note_request.text,
    )

    if len(cleaned_text) < 5:
        raise HTTPException(
            status_code=400,
            detail="Input text is too short",
        )

    if len(cleaned_text) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail="Input text is too large",
        )

    selected_format = validate_format(
        note_request.format,
    )

    legacy_task_map = {
        "bullet": "bullet_summary",
        "short": "short_summary",
        "detailed": "important_notes",
        "keypoints": "key_points",
        "beginner": "beginner_explanation",
        "qa": "qa_generation",
    }

    selected_task = legacy_task_map.get(
        selected_format,
        "bullet_summary",
    )

    selected_mode = (
        "student"
        if selected_task in {
            "important_notes",
            "qa_generation",
            "beginner_explanation",
        }
        else "general"
    )

    try:
        result = lumina_router.generate_from_text(
            text=cleaned_text,
            mode=selected_mode,
            task=selected_task,
        )

        formatted = result.get(
            "formatted_output",
            {},
        )

        summary = formatted.get(
            "markdown",
            "",
        )

        if not summary:
            error_detail = extract_generation_error(result)
            print("Legacy empty output details:", error_detail)

            raise HTTPException(
                status_code=500,
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
        )

        return JSONResponse(
            content={
                "summary": summary,
                "format": selected_format,
                "mode": selected_mode,
                "task": selected_task,
                "inputLength": len(cleaned_text),
                "usageCount": usage_count,
                "dailyLimit": DAILY_FREE_LIMIT,
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print("Legacy summarize via V2 pipeline error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary. Please try again.",
        )

