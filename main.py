from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import os
import re
import html
import json
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from datetime import datetime, timezone

from model_systems.pipeline_router import PipelineRouter


load_dotenv()

APP_NAME = "Lumina AI"
CONTACT_EMAIL = "support@lumina-ai.co.in"
BASE_URL = "https://www.lumina-ai.co.in"

OPENROUTER_API_KEY = os.getenv("LUMINA_API_KEY")

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

MAX_INPUT_LENGTH = int(
    os.getenv("MAX_INPUT_LENGTH", "45000")
)

DAILY_FREE_LIMIT = int(
    os.getenv("DAILY_FREE_LIMIT", "15")
)

MODEL_NAME = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct",
)

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY missing")


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
    allow_origins=[
        "http://localhost:5000",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "https://www.lumina-ai.co.in",
        "https://lumina-ai.co.in",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "authorization",
    ],
)


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
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
            background: linear-gradient(135deg, #eef2ff, #f8fafc);
            color: #1e293b;
            min-height: 100vh;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 920px;
            margin: auto;
            background: #ffffff;
            border-radius: 28px;
            padding: 52px;
            box-shadow: 0 12px 45px rgba(15, 23, 42, 0.08);
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
        "version": "2.0.0",
        "legacy_model": MODEL_NAME,
        "model_system_enabled": True,
    }


@app.get("/download-app", response_class=HTMLResponse)
def download_app():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Download Lumina AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; background:#f8fafc; color:#111827; min-height:100vh; display:flex; align-items:center; justify-content:center;">
    <main style="max-width:560px; background:white; padding:40px; border-radius:28px; text-align:center;">
        <h1>Download Lumina AI</h1>
        <p>Lumina AI helps you convert PDFs, images, camera scans, and notes into clean AI-powered study summaries.</p>
        <a style="display:inline-block; background:#4f46e5; color:white; text-decoration:none; padding:15px 28px; border-radius:16px; font-weight:700;" href="/static/lumina-ai.apk" download>
            Download APK
        </a>
        <p style="margin-top:22px; color:#64748b;">Play Store version coming soon.</p>
    </main>
</body>
</html>
"""


@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    body = f"""
        <p>
            Lumina AI respects user privacy and is designed to collect only the
            information required to provide authentication, saved history, and
            AI-powered summarization features.
        </p>

        <h2>Information We Collect</h2>
        <ul>
            <li>Name, username, and email address during account creation.</li>
            <li>Authentication data required for login and account access.</li>
            <li>Uploaded or pasted notes, extracted text, and generated summaries.</li>
            <li>Saved folders, favorites, and summary history linked to the user's account.</li>
        </ul>

        <h2>How We Use Information</h2>
        <ul>
            <li>To generate AI summaries and study notes.</li>
            <li>To save and organize user-specific summary history.</li>
            <li>To provide account-based access and security.</li>
            <li>To improve reliability, performance, and user experience.</li>
        </ul>

        <h2>AI Processing</h2>
        <p>
            Text submitted for summarization may be processed by AI model providers
            used by Lumina AI. Users should avoid uploading highly sensitive,
            confidential, or legally restricted content.
        </p>

        <h2>Third-Party Services</h2>
        <p>
            Lumina AI may use Firebase for authentication and database storage,
            Google services for sign-in where enabled, and OpenRouter, Gemini, Ollama,
            or compatible AI model providers for text processing.
        </p>

        <h2>Data Security</h2>
        <p>
            Lumina AI uses account-based storage and user identifiers to keep saved
            summaries separated between users. We do not sell user data.
        </p>

        <h2>User Control</h2>
        <p>
            Users may delete summaries from inside the app. Users may also delete
            their account and saved data from the Profile section.
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
            <li>Users should verify important academic, legal, medical, or professional information independently.</li>
        </ul>

        <h2>AI Generated Content</h2>
        <p>
            AI-generated summaries may contain mistakes, omissions, or imperfect
            interpretations. Lumina AI is a study assistant, not a replacement for
            professional, academic, legal, or medical advice.
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
            raise HTTPException(
                status_code=500,
                detail="AI returned empty output",
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


@app.post("/summarize")
@limiter.limit("20/minute")
def summarize_notes(
    request: Request,
    note_request: NoteRequest,
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

    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="AI provider API key is not configured",
        )

    cleaned_text = clean_input_text(note_request.text)

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

    selected_style = summary_styles.get(
        selected_format,
        summary_styles["bullet"],
    )

    prompt = f"""
You are Lumina AI, a precise academic note-cleaning and summarization assistant.

Your goal:
Transform the user's extracted notes into clean, accurate, useful study material.

Important behavior:
- Use ONLY the information present in the user's text.
- Do NOT invent facts, examples, formulas, names, dates, or explanations.
- If the text is messy because of OCR, silently clean it.
- Remove random OCR symbols, repeated characters, broken spacing, and meaningless fragments.
- Preserve mathematical formulas, definitions, technical terms, headings, and step-by-step processes.
- Keep the output readable on mobile.
- Do not include disclaimers, introductions, or phrases like "Here is".
- Do not mention OCR unless the user text itself is about OCR.
- Do not add markdown tables unless the content clearly needs comparison.
- If the input is too unclear, provide the cleanest possible structured extraction instead of guessing.

Selected output style:
{selected_style}

Formatting:
- Use clean markdown-style formatting.
- Use headings only when helpful.
- Use bullets for readability.
- Avoid random special characters.
- Avoid over-formatting.
- Keep line breaks clean.

User text:
\"\"\"
{cleaned_text}
\"\"\"
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.15,
            top_p=0.8,
            max_tokens=1800,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Lumina AI, a careful academic summarizer. "
                        "You clean OCR text, preserve facts, avoid hallucinations, "
                        "and output only useful study-ready content."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        summary = response.choices[0].message.content
        summary = clean_ai_output(summary)

        if not summary:
            raise HTTPException(
                status_code=500,
                detail="AI returned empty summary",
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
        )

        return JSONResponse(
            content={
                "summary": summary,
                "format": selected_format,
                "inputLength": len(cleaned_text),
                "usageCount": usage_count,
                "dailyLimit": DAILY_FREE_LIMIT,
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        error_message = str(e)

        if "No endpoints found" in error_message:
            raise HTTPException(
                status_code=502,
                detail="Selected AI model is unavailable. Please change the model.",
            )

        if "rate" in error_message.lower():
            raise HTTPException(
                status_code=429,
                detail="AI provider is busy. Please try again shortly.",
            )

        print("Legacy summarize error:", str(e))

        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary. Please try again.",
        )