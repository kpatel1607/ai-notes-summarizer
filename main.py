from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
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

from model_systems.pipeline_router import PipelineRouter


load_dotenv()

APP_NAME = "Lumina AI"
CONTACT_EMAIL = "support@lumina-ai.co.in"
BASE_URL = os.getenv(
    "BASE_URL",
    "https://www.lumina-ai.co.in",
).rstrip("/")
APP_VERSION_NAME = os.getenv("APP_VERSION_NAME", "2.0.0")
APP_VERSION_CODE = int(os.getenv("APP_VERSION_CODE", "2"))
APP_DOWNLOAD_PATH = os.getenv("APP_DOWNLOAD_PATH", "/static/Lumina-AI.apk")
APP_RELEASE_NOTES = [
    "Improved AI modes for student, professional, and general summaries.",
    "Cleaner OCR handling, structured folders, favorites, and document history.",
    "Updated privacy, terms, account deletion, and download pages.",
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
app.mount("/static", StaticFiles(directory="static"), name="static")


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


def app_download_url() -> str:
    if APP_DOWNLOAD_PATH.startswith("http"):
        return APP_DOWNLOAD_PATH

    return f"{BASE_URL}{APP_DOWNLOAD_PATH}"


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


def verify_firebase_user(
    authorization: str = Header(None),
):
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
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Inter, Arial, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            min-height: 100vh;
        }}

        .topbar {{
            position: sticky;
            top: 0;
            background: rgba(248, 250, 252, 0.94);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid #e2e8f0;
            z-index: 10;
        }}

        .nav {{
            max-width: 1080px;
            margin: auto;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }}

        .brand {{
            font-weight: 850;
            color: #111827;
            letter-spacing: .2px;
        }}

        .nav-links {{
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}

        .container {{
            max-width: 920px;
            margin: 38px auto;
            background: #ffffff;
            border-radius: 18px;
            padding: 52px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
        }}

        .badge {{
            display: inline-block;
            background: #eef2ff;
            color: #4f46e5;
            padding: 8px 16px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 14px;
            margin-bottom: 24px;
        }}

        h1 {{
            font-size: 42px;
            color: #111827;
            margin-bottom: 10px;
        }}

        .subtitle {{
            color: #64748b;
            margin-bottom: 36px;
            font-size: 16px;
        }}

        h2 {{
            margin-top: 34px;
            margin-bottom: 12px;
            color: #111827;
            font-size: 23px;
        }}

        p, li {{
            line-height: 1.8;
            color: #475569;
            font-size: 16px;
        }}

        ul {{
            margin-top: 12px;
            margin-left: 22px;
        }}

        li {{
            margin-bottom: 10px;
        }}

        a {{
            color: #4f46e5;
            font-weight: 600;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}

        .footer {{
            margin-top: 48px;
            padding-top: 22px;
            border-top: 1px solid #e2e8f0;
            color: #64748b;
            font-size: 14px;
        }}

        .notice {{
            margin: 28px 0;
            padding: 18px;
            border: 1px solid #c7d2fe;
            background: #eef2ff;
            border-radius: 14px;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 32px;
            }}

            h1 {{
                font-size: 34px;
            }}
        }}
    </style>
</head>
<body>
    <header class="topbar">
        <nav class="nav">
            <a class="brand" href="/">Lumina AI</a>
            <div class="nav-links">
                <a href="/download-app">Download</a>
                <a href="/privacy-policy">Privacy</a>
                <a href="/terms-and-conditions">Terms</a>
            </div>
        </nav>
    </header>
    <main class="container">
        <div class="badge">Lumina AI</div>
        <h1>{html.escape(title)}</h1>
        <p class="subtitle">{html.escape(subtitle)}</p>
        {body}
        <div class="footer">
            © 2026 Lumina AI. All rights reserved.
            <br>
            Contact: <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </div>
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
</head>
<body style="font-family: Arial, sans-serif; background:#f8fafc; color:#111827; padding:40px;">
    <main style="max-width:960px; margin:auto; background:white; padding:48px; border-radius:28px;">
        <h1>Lumina AI</h1>
        <p>
            Lumina AI converts notes, PDFs, scanned pages, images, and camera-captured text
            into clean AI-powered summaries, study notes, key points, beginner explanations,
            and question-answer formats.
        </p>

        <h2>API Status</h2>
        <p>Backend is running successfully.</p>

        <p>
            <a href="/health">Health Check</a> |
            <a href="/privacy-policy">Privacy Policy</a> |
            <a href="/terms-and-conditions">Terms & Conditions</a>
        </p>
    </main>
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


@app.get("/download-app", response_class=HTMLResponse)
def download_app():
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Download Lumina AI</title>
    <meta name="description" content="Download the latest Lumina AI Android app.">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: Inter, Arial, sans-serif;
            background: #f8fafc;
            color: #111827;
        }}
        .wrap {{
            max-width: 1040px;
            margin: auto;
            padding: 44px 20px;
        }}
        .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) minmax(280px, .8fr);
            gap: 28px;
            align-items: center;
        }}
        .panel {{
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 34px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, .06);
        }}
        h1 {{ font-size: clamp(34px, 6vw, 58px); line-height: 1; margin: 0 0 18px; }}
        h2 {{ margin: 0 0 14px; }}
        p, li {{ color: #475569; line-height: 1.7; }}
        .button {{
            display: inline-block;
            background: #4f46e5;
            color: white;
            text-decoration: none;
            padding: 15px 24px;
            border-radius: 12px;
            font-weight: 800;
            margin-top: 12px;
        }}
        .muted {{ color: #64748b; font-size: 14px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-top: 24px;
        }}
        .card {{
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 20px;
        }}
        @media (max-width: 780px) {{
            .hero, .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <main class="wrap">
        <section class="hero">
            <div>
                <p class="muted">Latest Android release</p>
                <h1>Lumina AI</h1>
                <p>
                    Convert notes, PDFs, camera scans, and images into structured summaries,
                    Q&A, flashcards, revision sheets, and professional reports.
                </p>
                <a class="button" href="{APP_DOWNLOAD_PATH}" download>Download APK v{APP_VERSION_NAME}</a>
                <p class="muted">When the Play Store listing is live, this button can point to the official store URL.</p>
            </div>
            <aside class="panel">
                <h2>Release Notes</h2>
                <ul>
                    {''.join(f'<li>{html.escape(note)}</li>' for note in APP_RELEASE_NOTES)}
                </ul>
                <p class="muted">Version code: {APP_VERSION_CODE}</p>
            </aside>
        </section>
        <section class="grid">
            <div class="card"><strong>Private accounts</strong><p>Email/Google sign-in, account deletion, and saved history controls.</p></div>
            <div class="card"><strong>Organized documents</strong><p>Folders, favorites, search, and generated-document history.</p></div>
            <div class="card"><strong>AI modes</strong><p>Student, professional, and general outputs tuned for different workflows.</p></div>
        </section>
    </main>
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
            Some security options, such as password reset, email verification,
            recovery email records, and multi-factor authentication, depend on
            Firebase and platform availability.
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

