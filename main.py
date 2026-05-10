from fastapi import (
    FastAPI,
    Header,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    JSONResponse,
    HTMLResponse,
)

from pydantic import BaseModel

from openai import OpenAI

from dotenv import load_dotenv

import os


# =========================
# LOAD ENV
# =========================

load_dotenv()


# =========================
# APP CONFIG
# =========================

app = FastAPI(

    title="Lumina AI API",

    description="""
    AI-powered academic summarization backend for Lumina.
    """,

    version="1.0.0",
)


# =========================
# CORS
# =========================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# =========================
# ENV VARIABLES
# =========================

API_KEY = os.getenv("API_KEY")

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

if not API_KEY:
    print("WARNING: API_KEY missing")

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY missing")


# =========================
# OPENROUTER CLIENT
# =========================

client = OpenAI(

    api_key=OPENROUTER_API_KEY,

    base_url=
        "https://openrouter.ai/api/v1",
)


# =========================
# REQUEST MODEL
# =========================

class NoteRequest(BaseModel):

    text: str

    format: str = "bullet"


# =========================
# API KEY VERIFICATION
# =========================

def verify_api_key(
    x_api_key: str = Header(...)
):

    if x_api_key != API_KEY:

        raise HTTPException(

            status_code=403,

            detail="Invalid API Key",
        )


# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {

        "app": "Lumina AI",

        "status": "running",

        "message":
            "Lumina AI backend is live",
    }


# =========================
# HEALTH CHECK
# =========================

@app.get("/health")
def health_check():

    return {

        "status": "healthy",
    }


# =========================
# PRIVACY POLICY
# =========================

@app.get(
    "/privacy-policy",
    response_class=HTMLResponse,
)
def privacy_policy():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>
        Lumina Privacy Policy
    </title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #eef2ff,
                    #f8fafc
                );

            color: #1e293b;

            min-height: 100vh;

            padding: 40px 20px;
        }

        .container {

            max-width: 900px;

            margin: auto;

            background: white;

            border-radius: 24px;

            padding: 50px;

            box-shadow:
                0 10px 40px rgba(
                    0,
                    0,
                    0,
                    0.08
                );
        }

        h1 {

            font-size: 42px;

            color: #4f46e5;

            margin-bottom: 10px;
        }

        .subtitle {

            color: #64748b;

            margin-bottom: 40px;

            font-size: 16px;
        }

        h2 {

            margin-top: 35px;

            margin-bottom: 15px;

            color: #111827;

            font-size: 24px;
        }

        p {

            line-height: 1.8;

            color: #475569;

            font-size: 16px;
        }

        ul {

            margin-top: 15px;
            margin-left: 20px;
        }

        li {

            margin-bottom: 12px;

            line-height: 1.7;

            color: #475569;
        }

        .footer {

            margin-top: 50px;

            padding-top: 20px;

            border-top:
                1px solid #e2e8f0;

            color: #64748b;

            font-size: 14px;
        }

    </style>

</head>

<body>

    <div class="container">

        <h1>
            Privacy Policy
        </h1>

        <p class="subtitle">
            Last updated: 2026
        </p>

        <p>
            Lumina respects your privacy and is committed
            to protecting your personal information.
        </p>

        <h2>
            Information We Collect
        </h2>

        <ul>

            <li>
                Email address
            </li>

            <li>
                Name and username
            </li>

            <li>
                AI-generated summaries
            </li>

            <li>
                Authentication information
            </li>

        </ul>

        <h2>
            How We Use Information
        </h2>

        <p>
            We use your data to provide AI-powered
            summarization services, improve app
            performance, maintain security, and
            personalize user experience.
        </p>

        <h2>
            Data Protection
        </h2>

        <p>
            Your information is securely stored and
            protected using industry-standard security
            practices.
        </p>

        <h2>
            Third-Party Services
        </h2>

        <p>
            Lumina may use services such as Firebase,
            OpenRouter, and Google Authentication
            for core functionality.
        </p>

        <h2>
            Contact
        </h2>

        <p>
            Email:
            lumina.ai.app@gmail.com
        </p>

        <div class="footer">

            © 2026 Lumina AI.
            All rights reserved.

        </div>

    </div>

</body>

</html>
"""


# =========================
# TERMS & CONDITIONS
# =========================

@app.get(
    "/terms-and-conditions",
    response_class=HTMLResponse,
)
def terms_and_conditions():

    return """
<!DOCTYPE html>

<html>

<head>

    <title>
        Lumina Terms & Conditions
    </title>

    <meta name="viewport"
          content="width=device-width, initial-scale=1.0">

    <style>

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {

            font-family: Arial, sans-serif;

            background:
                linear-gradient(
                    135deg,
                    #eef2ff,
                    #f8fafc
                );

            color: #1e293b;

            min-height: 100vh;

            padding: 40px 20px;
        }

        .container {

            max-width: 900px;

            margin: auto;

            background: white;

            border-radius: 24px;

            padding: 50px;

            box-shadow:
                0 10px 40px rgba(
                    0,
                    0,
                    0,
                    0.08
                );
        }

        h1 {

            font-size: 42px;

            color: #4f46e5;

            margin-bottom: 10px;
        }

        .subtitle {

            color: #64748b;

            margin-bottom: 40px;

            font-size: 16px;
        }

        h2 {

            margin-top: 35px;

            margin-bottom: 15px;

            color: #111827;

            font-size: 24px;
        }

        p {

            line-height: 1.8;

            color: #475569;

            font-size: 16px;
        }

        ul {

            margin-top: 15px;
            margin-left: 20px;
        }

        li {

            margin-bottom: 12px;

            line-height: 1.7;

            color: #475569;
        }

        .footer {

            margin-top: 50px;

            padding-top: 20px;

            border-top:
                1px solid #e2e8f0;

            color: #64748b;

            font-size: 14px;
        }

    </style>

</head>

<body>

    <div class="container">

        <h1>
            Terms & Conditions
        </h1>

        <p class="subtitle">
            Last updated: 2026
        </p>

        <p>
            By accessing and using Lumina,
            you agree to the following terms.
        </p>

        <h2>
            Use of Service
        </h2>

        <p>
            Lumina provides AI-powered note
            summarization tools intended for
            educational and productivity use.
        </p>

        <h2>
            User Responsibilities
        </h2>

        <ul>

            <li>
                Users must not upload harmful,
                illegal, or abusive content.
            </li>

            <li>
                Users are responsible for
                their generated outputs.
            </li>

            <li>
                Abuse of the platform may
                result in account suspension.
            </li>

        </ul>

        <h2>
            AI Generated Content
        </h2>

        <p>
            AI-generated summaries may occasionally
            contain inaccuracies. Users should
            verify important information independently.
        </p>

        <h2>
            Account Security
        </h2>

        <p>
            Users are responsible for maintaining
            the security of their accounts
            and passwords.
        </p>

        <h2>
            Changes to Terms
        </h2>

        <p>
            Lumina reserves the right to modify
            these terms at any time.
        </p>

        <h2>
            Contact
        </h2>

        <p>
            Email:
            lumina.ai.app@gmail.com
        </p>

        <div class="footer">

            © 2026 Lumina AI.
            All rights reserved.

        </div>

    </div>

</body>

</html>
"""


# =========================
# SUMMARY STYLES
# =========================

summary_styles = {

    "bullet": """

Create structured bullet-point notes.

Requirements:
- Use concise bullet points
- Keep only important information
- Remove repetition
- Preserve definitions, formulas, facts, and concepts
- Use sub-bullets when necessary
- Make it suitable for revision

""",

    "short": """

Create a very concise summary.

Requirements:
- Maximum clarity in minimum words
- Use short paragraphs
- Focus only on core ideas
- Remove examples and unnecessary details

""",

    "detailed": """

Create detailed study notes.

Requirements:
- Explain concepts clearly
- Preserve important details
- Use headings and subheadings
- Keep information well-structured
- Include examples if present
- Make it useful for exam preparation

""",

    "keypoints": """

Extract the most important key points only.

Requirements:
- Focus on critical concepts
- Keep output compact
- Prioritize formulas, definitions, and facts

""",

    "beginner": """

Explain the content for a beginner.

Requirements:
- Use simple language
- Break down difficult concepts
- Avoid complex jargon
- Make learning intuitive

""",

    "qa": """

Convert notes into question-answer format.

Requirements:
- Generate meaningful questions
- Provide concise answers
- Cover major concepts
- Keep formatting clean

""",
}


# =========================
# SUMMARIZE ENDPOINT
# =========================

@app.post("/summarize")
def summarize_notes(

    request: NoteRequest,

    x_api_key: str = Header(None),
):

    verify_api_key(x_api_key)

    try:

        selected_style = summary_styles.get(

            request.format,

            summary_styles["bullet"],
        )

        prompt = f"""

You are an expert AI academic summarizer.

Your task is to generate clean,
high-quality educational summaries.

{selected_style}

Global Rules:

- No introductions
- No conclusions
- No filler sentences
- Maintain factual accuracy
- Preserve formulas and definitions
- Preserve sequence when needed
- Use clean formatting
- Avoid repetition

Text:
{request.text}

"""

        response = client.chat.completions.create(

            model=
                "meta-llama/llama-3.1-8b-instruct",

            temperature=0.3,

            messages=[

                {
                    "role": "system",

                    "content":
                        "You are an intelligent academic summarizer.",
                },

                {
                    "role": "user",

                    "content": prompt,
                },
            ],
        )

        summary = (
            response
            .choices[0]
            .message
            .content
        )

        return JSONResponse(

            content={
                "summary": summary,
            }
        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e),
        )
