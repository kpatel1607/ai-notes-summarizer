from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

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

load_dotenv()

APP_NAME = "Lumina AI"
CONTACT_EMAIL = "support@lumina-ai.co.in"
BASE_URL = "https://www.lumina-ai.co.in"



# API_KEY = os.getenv("API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

# if not API_KEY:
#     print("WARNING: API_KEY missing")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY missing")


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["120/minute"],
)

app = FastAPI(
    title="Lumina AI API",
    description="AI-powered OCR cleanup and academic summarization backend for Lumina.",
    version="1.2.0",
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
    allow_origins=["http://localhost:5000",
                   "https://www.lumina-ai.co.in",
                   "https://lumina-ai.co.in"
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


# def verify_api_key(
#     x_api_key: str = Header(None),
# ):
    # if not API_KEY:
    #     raise HTTPException(
    #         status_code=500,
    #         detail="Server API key is not configured",
    #     )
    #
    # if not x_api_key or x_api_key != API_KEY:
    #     raise HTTPException(
    #         status_code=403,
    #         detail="Invalid API Key")
    #

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

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: Inter, Arial, sans-serif;
            background: linear-gradient(135deg, #eef2ff, #f8fafc);
            color: #111827;
            min-height: 100vh;
            padding: 32px 18px;
        }}

        .container {{
            max-width: 1080px;
            margin: auto;
            background: #ffffff;
            border-radius: 32px;
            padding: 64px;
            box-shadow: 0 14px 50px rgba(15, 23, 42, 0.08);
        }}

        .badge {{
            display: inline-block;
            padding: 8px 16px;
            background: #eef2ff;
            color: #4f46e5;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 24px;
        }}

        h1 {{
            font-size: 56px;
            line-height: 1.1;
            margin-bottom: 22px;
        }}

        .highlight {{
            color: #4f46e5;
        }}

        .hero-text {{
            font-size: 19px;
            line-height: 1.8;
            color: #475569;
            max-width: 820px;
        }}

        .section-title {{
            margin-top: 46px;
            font-size: 28px;
            font-weight: 800;
        }}

        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 18px;
            margin-top: 22px;
        }}

        .feature {{
            background: #f8fafc;
            padding: 22px;
            border-radius: 20px;
            border: 1px solid #e5e7eb;
        }}

        .feature strong {{
            display: block;
            color: #111827;
            font-size: 17px;
            margin-bottom: 8px;
        }}

        .feature p {{
            color: #475569;
            line-height: 1.7;
            font-size: 15px;
        }}

        .info {{
            margin-top: 28px;
            padding: 24px;
            border-radius: 22px;
            background: #eef2ff;
            color: #3730a3;
            line-height: 1.8;
        }}

        .buttons {{
            margin-top: 40px;
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
        }}

        .btn {{
            text-decoration: none;
            padding: 15px 28px;
            border-radius: 15px;
            font-weight: 800;
        }}

        .primary {{
            background: #4f46e5;
            color: white;
        }}

        .secondary {{
            background: #eef2ff;
            color: #4f46e5;
        }}

        .footer {{
            margin-top: 56px;
            padding-top: 24px;
            border-top: 1px solid #e5e7eb;
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            color: #64748b;
            font-size: 14px;
        }}

        .footer a {{
            color: #4f46e5;
            text-decoration: none;
            font-weight: 700;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 34px;
            }}

            h1 {{
                font-size: 38px;
            }}

            .hero-text {{
                font-size: 16px;
            }}
        }}
    </style>
</head>

<body>
    <main class="container">
        <div class="badge">AI Powered Study Assistant</div>

        <h1>
            Lumina AI helps you turn
            <span class="highlight">notes into study-ready summaries</span>.
        </h1>

        <p class="hero-text">
            Lumina AI is an educational productivity app that helps users convert
            raw notes, PDFs, scanned pages, images, and camera-captured text into
            clean AI-generated summaries, revision notes, key points, beginner
            explanations, and question-answer study formats.
        </p>

        <h2 class="section-title">What Lumina AI Does</h2>

        <section class="features">
            <div class="feature">
                <strong>PDF Note Processing</strong>
                <p>
                    Users can upload PDF study material and extract text for AI-powered summarization.
                </p>
            </div>

            <div class="feature">
                <strong>Image & Camera OCR</strong>
                <p>
                    Lumina can extract text from uploaded images and camera scans of handwritten or printed notes.
                </p>
            </div>

            <div class="feature">
                <strong>AI Summaries</strong>
                <p>
                    The app generates clean summaries, bullet notes, detailed notes, key points, beginner explanations, and Q&A formats.
                </p>
            </div>

            <div class="feature">
                <strong>Saved History</strong>
                <p>
                    Logged-in users can save summaries, organize them into folders, mark favorites, and access history securely.
                </p>
            </div>

            <div class="feature">
                <strong>Export & Share</strong>
                <p>
                    Users can copy, share, and export generated study summaries for later revision.
                </p>
            </div>

            <div class="feature">
                <strong>Account Control</strong>
                <p>
                    Users can manage their account, logout, and delete their account and saved data from inside the app.
                </p>
            </div>
        </section>

        <div class="info">
            Lumina AI is designed for students, learners, and productivity-focused users.
            It is not a replacement for professional, medical, legal, or academic advice.
            Users should verify important information independently.
        </div>

        <div class="buttons">
            <a href="/download-app" class="btn primary">Download App</a>
            <a href="/privacy-policy" class="btn secondary">Privacy Policy</a>
            <a href="/terms-and-conditions" class="btn secondary">Terms & Conditions</a>
        </div>

        <div class="footer">
            <div>
                © 2026 Lumina AI. All rights reserved.
                <br>
                Support: <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
            </div>

            <div>
                <a href="/privacy-policy">Privacy Policy</a>
                &nbsp; | &nbsp;
                <a href="/terms-and-conditions">Terms & Conditions</a>
            </div>
        </div>
    </main>
</body>
</html>
"""


@app.get("/health")
def health_check():
    return {
        "app": APP_NAME,
        "status": "healthy",
        "version": "1.2.0",
        "model": MODEL_NAME,
    }

@app.get("/download-app", response_class=HTMLResponse)
def download_app():
    return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Download Lumina AI</title>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f8fafc;
                    color: #111827;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 24px;
                }
        
                .card {
                    max-width: 560px;
                    background: white;
                    padding: 40px;
                    border-radius: 28px;
                    box-shadow: 0 12px 40px rgba(15, 23, 42, 0.08);
                    text-align: center;
                }
        
                h1 {
                    font-size: 34px;
                    margin-bottom: 14px;
                }
        
                p {
                    color: #475569;
                    line-height: 1.7;
                    margin-bottom: 28px;
                }
        
                .btn {
                    display: inline-block;
                    background: #4f46e5;
                    color: white;
                    text-decoration: none;
                    padding: 15px 28px;
                    border-radius: 16px;
                    font-weight: 700;
                }
        
                .note {
                    margin-top: 22px;
                    font-size: 14px;
                    color: #64748b;
                }
            </style>
        </head>
        <body>
            <main class="card">
                <h1>Download Lumina AI</h1>
                <p>
                    Lumina AI helps you convert PDFs, images, camera scans,
                    and notes into clean AI-powered study summaries.
                </p>
        
                <a class="btn" href="/static/lumina-ai.apk" download>
                    Download APK
                </a>
        
                <p class="note">
                    Play Store version coming soon.
                </p>
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
            Google services for sign-in where enabled, and OpenRouter or compatible
            AI model providers for text processing.
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

        check_and_increment_daily_usage(
            user_uid,
        )

        return JSONResponse(
            content={
                "summary": summary,
                "format": selected_format,
                "inputLength": len(cleaned_text),
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

        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary. Please try again.",
        )
