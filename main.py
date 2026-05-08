from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("API_KEY")

if not API_KEY or not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: API_KEY missing")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# OpenRouter setup
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Request model
class NoteRequest(BaseModel):
    text: str
    format: str = "bullet"

@app.get("/")
def home():
    return {"message": "AI Notes Summarizer API is running"}

@app.post("/summarize")
def summarize_notes(request: NoteRequest, x_api_key: str = Header(None)):

    verify_api_key(x_api_key)

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
    - Include examples if present in original text
    - Make it useful for exam preparation
    """,

        "keypoints": """
    Extract the most important key points only.

    Requirements:
    - Focus on critical concepts
    - Keep output compact
    - Prioritize facts, formulas, and definitions
    - Avoid explanations unless necessary
    """,

        "beginner": """
    Explain the content for a complete beginner.

    Requirements:
    - Use very simple language
    - Break down difficult concepts
    - Avoid technical jargon when possible
    - Make explanations intuitive and easy to understand
    - Keep learning-friendly formatting
    """,

        "qa": """
    Convert the notes into study question-answer format.

    Requirements:
    - Generate meaningful questions
    - Provide concise accurate answers
    - Cover all major concepts
    - Keep formatting clean and readable
    """
    }

    selected_style = summary_styles.get(
        request.format,
        summary_styles["bullet"]
    )

    prompt = f"""
    You are an expert AI academic summarizer and study assistant.

    Your task is to analyze the provided notes carefully and generate high-quality educational summaries.

    {selected_style}

    Global Rules:
    - Do NOT add introductions or conclusions
    - Do NOT say phrases like:
      "Here is the summary"
      "Sure"
      "I can help with that"
    - Output ONLY the processed content
    - Maintain factual accuracy
    - Preserve important technical information
    - Preserve chronological order when relevant
    - Handle long multi-paragraph text correctly
    - Ignore irrelevant filler sentences
    - Make formatting visually clean
    - Use markdown-style formatting when useful
    - Avoid repeating the same idea multiple times

    If the input contains:
    - formulas → preserve them
    - definitions → preserve them clearly
    - steps/processes → keep sequential order
    - comparisons → preserve comparison structure

    Text to process:
    {request.text}
    """

    response = client.chat.completions.create(
        model="google/gemma-2-9b-it:free",
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent academic summarizer."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    summary = response.choices[0].message.content

    return JSONResponse(content={"summary": summary})
