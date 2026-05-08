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

@app.get("/")
def home():
    return {"message": "AI Notes Summarizer API is running"}

@app.post("/summarize")
def summarize_notes(request: NoteRequest, x_api_key: str = Header(None)):

    verify_api_key(x_api_key)

    prompt = f"""
                Summarize the following text in simple bullet points.
                Do NOT add any introduction like "Here is a summary".
                Do NOT explain what you are doing.
                Only output the summary.
                
                Text:
                {request.text}
                """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a study assistant AI."},
            {"role": "user", "content": prompt}
        ]
    )

    summary = response.choices[0].message.content

    return JSONResponse(content={"summary": summary})