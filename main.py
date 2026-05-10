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
    <html>
    <head>
        <title>Lumina Privacy Policy</title>

        <style>

            body {
                font-family: Arial;
                max-width: 850px;
                margin: auto;
                padding: 40px;
                line-height: 1.7;
                background: #f8f9fc;
                color: #222;
            }

            h1 {
                color: #4f46e5;
            }

        </style>
    </head>

    <body>

        <h1>Privacy Policy</h1>

        <p>
            Lumina collects user information such as
            email address, username, and notes
            to provide AI-powered summarization services.
        </p>

        <p>
            User data is securely stored and is never sold
            to third parties.
        </p>

        <p>
            By using Lumina, you agree to the collection
            and processing of data necessary for
            authentication and AI summarization.
        </p>

        <p>
            Contact:
            lumina.ai.app@gmail.com
        </p>

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
    <html>

    <head>

        <title>
            Lumina Terms & Conditions
        </title>

        <style>

            body {
                font-family: Arial;
                max-width: 850px;
                margin: auto;
                padding: 40px;
                line-height: 1.7;
                background: #f8f9fc;
                color: #222;
            }

            h1 {
                color: #4f46e5;
            }

        </style>

    </head>

    <body>

        <h1>Terms & Conditions</h1>

        <p>
            By using Lumina, you agree to use the
            application responsibly and lawfully.
        </p>

        <p>
            AI-generated summaries may occasionally
            contain inaccuracies.
        </p>

        <p>
            Users are responsible for any uploaded
            content and generated outputs.
        </p>

        <p>
            Lumina reserves the right to update
            these terms at any time.
        </p>

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
