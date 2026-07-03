from fastapi import (
    BackgroundTasks,
    FastAPI,
    Header,
    HTTPException,
    Request,
    UploadFile,
    File,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import (
    JSONResponse,
    HTMLResponse,
    FileResponse,
    RedirectResponse,
    Response,
)
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
import time
import hashlib
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth, firestore
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from urllib.parse import urlparse

from model_systems.pipeline_router import PipelineRouter
from model_systems.job_status import JobStatusStore
from model_systems.redis_cache import RedisExactCache
from model_systems.request_hashing import RequestHasher
from model_systems.routing_logger import RoutingLogger
from model_systems.safety_controls import SafetyControls, SafetyError
from model_systems.schema_validator import SchemaValidator
from model_systems.semantic_cache import SemanticCache
from model_systems.task_queue import TaskQueue


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
APP_VERSION_NAME = os.getenv("PUBLIC_APP_VERSION_NAME", "3.0.0")
APP_VERSION_CODE = int(os.getenv("PUBLIC_APP_VERSION_CODE", "30"))
APP_DOWNLOAD_PATH = os.getenv("APP_DOWNLOAD_PATH", "/download-apk")
APK_FILE_PATH = os.getenv("APK_FILE_PATH", "static/Lumina-AI.apk")
APP_RELEASE_NOTES = [
    "Fixed login, signup, and Google sign-in loading states so auth screens recover after slow network or Firebase delays.",
    "Added auth timeouts for Firebase, Google, and Firestore profile sync calls.",
    "Improved login profile sync resilience by safely merging missing user documents.",
    "Fixed a pre-request generation hang so Generate Document reaches the backend reliably after extraction.",
    "Added safer generation cleanup so loading state, stop state, and screen-awake state reset after success, failure, or cancellation.",
    "Removed temporary frontend debug output from Firebase token retrieval.",
    "Added a stop button so users can cancel an in-progress generation from the workspace.",
    "Kept generation alive while users move between app sections, with screen-awake support during active generation.",
    "Added backend request diagnostics so generation starts, auth, routing, and completion are visible in server logs.",
    "Fixed light appearance combinations with Midnight and other styles so button, option, and card text remains readable.",
    "Added a cleaner Play Store-ready source picker with full extracted text review before generation.",
    "Added extraction quality badges for text, PDF, image OCR, and camera scan sources.",
    "Improved text, dropdown, button, and chip visibility across all appearance styles.",
    "Fixed email draft output so empty model templates are repaired from the original source text.",
    "Verified all Student, Professional, and General output API formats through the backend router.",
    "Improved export cache handling so it no longer depends on a hardcoded package path.",
    "Kept workspace contrast, animation, table, and output formatting improvements.",
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    print("Unhandled API error:", exc.__class__.__name__)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Unexpected server error. Please try again.",
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
request_hasher = RequestHasher()
exact_cache = RedisExactCache()
semantic_cache = SemanticCache()
job_status_store = JobStatusStore()
task_queue = TaskQueue()
schema_validator = SchemaValidator()
safety_controls = SafetyControls()


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


class FeedbackRequest(BaseModel):
    title: str = Field(default="", max_length=160)
    mode: str = Field(default="general", max_length=30)
    task: str = Field(default="short_summary", max_length=50)
    rating: int = Field(..., ge=1, le=5)
    tags: list[str] = Field(default_factory=list, max_length=12)
    route: str = Field(default="", max_length=40)
    modelTier: str = Field(default="", max_length=40)


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
        print("Firebase token verification error:", e.__class__.__name__)

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired login session",
        )


firestore_db = None

if firebase_admin._apps:
    firestore_db = firestore.client()

routing_logger = RoutingLogger(firestore_db)


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


def generation_error_status(result: Dict[str, Any]) -> int:
    generation_result = result.get("generation_result", {})

    if not isinstance(generation_result, dict):
        return 500

    error_type = generation_result.get("error_type", "")
    error = str(generation_result.get("error", "")).lower()

    if error_type == "quota_exceeded" or "quota" in error:
        return 429

    if error_type in {"timeout", "provider_error", "unexpected_error"}:
        return 503

    return 500


def extract_route_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    pipeline_output = result.get("pipeline_output", {})

    if not isinstance(pipeline_output, dict):
        return {
            "routeConfig": {},
            "complexity": {},
        }

    routing = pipeline_output.get("routing", {})

    if not isinstance(routing, dict):
        return {
            "routeConfig": {},
            "complexity": {},
        }

    return {
        "routeConfig": routing.get("route_config", {}),
        "complexity": routing.get("complexity", {}),
        "features": routing.get("features", {}),
        "input_analysis": routing.get("input_analysis", {}),
    }


def anonymized_request_id(
    *,
    user_uid: str,
    selected_mode: str,
    selected_task: str,
    cleaned_text: str,
) -> str:
    return request_hasher.cache_key(
        user_id=user_uid,
        task=selected_task,
        mode=selected_mode,
        text=cleaned_text,
    ).split(":")[-1]


def anonymized_file_request_id(file_path: str) -> str:
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def cached_payload_matches_request(
    cached_payload: Optional[Dict[str, Any]],
    *,
    selected_mode: str,
    selected_task: str,
) -> bool:
    if not isinstance(cached_payload, dict):
        return False

    cached_mode = str(cached_payload.get("mode") or "").lower().strip()
    cached_task = str(cached_payload.get("task") or "").lower().strip()

    return cached_mode == selected_mode and cached_task == selected_task


def log_routing_event(
    *,
    user_uid: str,
    request_id: str,
    route_metadata: Dict[str, Any],
    started_at: float,
    cache_hit: bool = False,
    error_type: str = "",
    user_feedback_rating: Optional[int] = None,
) -> None:
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    routing_logger.log(
        user_id=user_uid,
        anonymized_request_id=request_id,
        route_metadata=route_metadata,
        processing_time_ms=elapsed_ms,
        cache_hit=cache_hit,
        error_type=error_type,
        user_feedback_rating=user_feedback_rating,
    )


def usage_weight_for_route(route_metadata: Dict[str, Any]) -> int:
    route_config = route_metadata.get("routeConfig", {})
    complexity = route_metadata.get("complexity", {})
    score = int(complexity.get("score") or 0)

    if route_config.get("path") == "heavy_path" or score >= 62:
        return 2

    return 1


def raise_safety_http_error(error: SafetyError) -> None:
    status_code = 429

    if error.error_type in {"file_page_limit"}:
        status_code = 413
    elif error.error_type in {"heavy_requires_queue"}:
        status_code = 409
    elif error.error_type in {"processing_timeout"}:
        status_code = 504

    raise HTTPException(
        status_code=status_code,
        detail=error.message,
    )


def enforce_endpoint_safety(
    *,
    user_uid: str,
    client_ip: str,
    endpoint: str,
) -> None:
    try:
        safety_controls.enforce_endpoint_limit(
            user_id=user_uid,
            ip=client_ip,
            endpoint=endpoint,
        )
    except SafetyError as error:
        raise_safety_http_error(error)


def enforce_heavy_safety(
    *,
    user_uid: str,
    client_ip: str,
    route_metadata: Dict[str, Any],
    allow_queue: bool,
) -> None:
    try:
        safety_controls.enforce_heavy_limit(
            user_id=user_uid,
            ip=client_ip,
            route_metadata=route_metadata,
            allow_queue=allow_queue,
        )
    except SafetyError as error:
        raise_safety_http_error(error)


def build_generate_response_payload(
    *,
    formatted: Dict[str, Any],
    selected_mode: str,
    selected_task: str,
    usage_count: int,
    cache_hit: bool = False,
    cache_type: str = "",
    route_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_metadata = route_metadata or {}
    route_config = route_metadata.get("routeConfig", {})
    route = route_config.get("path", "")
    model_tier = route_config.get("model_tier", "")
    payload = {
        "success": True,
        "title": formatted.get("title", ""),
        "markdown": formatted.get("markdown", ""),
        "plainText": formatted.get("plainText") or formatted.get("plain_text", ""),
        "sections": formatted.get("sections", []),
        "sectionCount": formatted.get("sectionCount") or formatted.get("section_count", 0),
        "mode": formatted.get("mode", selected_mode),
        "task": formatted.get("task", selected_task),
        "format": formatted.get("format", ""),
        "provider": formatted.get("provider", ""),
        "model": formatted.get("model", ""),
        "route": formatted.get("route") or route,
        "modelTier": formatted.get("modelTier") or model_tier,
        "cached": cache_hit,
        "usageCount": usage_count,
        "dailyLimit": DAILY_FREE_LIMIT,
        "cacheHit": cache_hit,
        "cacheType": cache_type,
        **route_metadata,
    }
    return schema_validator.validate_generation_response(payload)


def generate_text_response_payload(
    *,
    user_uid: str,
    cleaned_text: str,
    selected_mode: str,
    selected_task: str,
    prepared: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prepared = prepared or lumina_router.process_text_for_mode(
        text=cleaned_text,
        mode=selected_mode,
        task=selected_task,
    )

    result = lumina_router.generate_prepared(
        prepared=prepared,
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
        raise RuntimeError(extract_generation_error(result))

    route_metadata = extract_route_metadata(result)
    usage_count = check_and_increment_daily_usage(
        user_uid,
        weight=usage_weight_for_route(route_metadata),
    )

    return build_generate_response_payload(
        formatted=formatted,
        selected_mode=selected_mode,
        selected_task=selected_task,
        usage_count=usage_count,
        route_metadata=route_metadata,
    )


def process_generation_job(
    *,
    job_id: str,
    user_uid: str,
    cleaned_text: str,
    selected_mode: str,
    selected_task: str,
    cache_key: str,
    request_id: str,
) -> None:
    started_at = time.perf_counter()
    route_metadata: Dict[str, Any] = {}

    try:
        job_status_store.update(
            job_id,
            status="processing",
        )

        prepared = lumina_router.process_text_for_mode(
            text=cleaned_text,
            mode=selected_mode,
            task=selected_task,
        )
        route_metadata = extract_route_metadata(prepared)

        cached_payload = exact_cache.get(cache_key)

        if cached_payload_matches_request(
            cached_payload,
            selected_mode=selected_mode,
            selected_task=selected_task,
        ):
            cached_payload["cacheHit"] = True
            cached_payload["cached"] = True
            cached_payload["cacheType"] = "exact"
            cached_payload["usageCount"] = get_daily_usage_count(user_uid)
            cached_payload["dailyLimit"] = DAILY_FREE_LIMIT
            cached_payload.update(route_metadata)
            cached_payload = schema_validator.validate_generation_response(
                cached_payload,
            )
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=True,
            )

            job_status_store.update(
                job_id,
                status="completed",
                result=cached_payload,
                route_metadata=route_metadata,
            )
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=True,
            )
            return

        response_payload = generate_text_response_payload(
            user_uid=user_uid,
            cleaned_text=cleaned_text,
            selected_mode=selected_mode,
            selected_task=selected_task,
            prepared=prepared,
        )

        exact_cache.set(cache_key, response_payload)
        semantic_cache.save_embedding_cache(
            text=cleaned_text,
            task=selected_task,
            mode=selected_mode,
            response=response_payload,
            metadata=route_metadata,
        )

        job_status_store.update(
            job_id,
            status="completed",
            result=response_payload,
            route_metadata=route_metadata,
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
        )
    except HTTPException as exc:
        job_status_store.update(
            job_id,
            status="failed",
            error=str(exc.detail),
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=str(exc.status_code),
        )
    except Exception as exc:
        print("Generation job failed:", exc.__class__.__name__)
        job_status_store.update(
            job_id,
            status="failed",
            error="Failed to generate output. Please try again.",
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=exc.__class__.__name__,
        )


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
    *,
    description: str,
    canonical_path: str,
    heading: str | None = None,
) -> str:
    canonical_url = f"{BASE_URL}{canonical_path}"
    page_heading = heading or title

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <title>{html.escape(title)}</title>
    <meta name="description" content="{html.escape(description)}">
    <link rel="canonical" href="{html.escape(canonical_url)}">
    <meta property="og:title" content="{html.escape(title)}">
    <meta property="og:description" content="{html.escape(description)}">
    <meta property="og:url" content="{html.escape(canonical_url)}">
    <meta property="og:type" content="website">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {site_styles()}
</head>
<body>
    {site_nav("/")}
    <main class="legal-layout">
      <article class="legal-card">
        <span class="eyebrow">Lumina AI legal</span>
        <h1>{html.escape(page_heading)}</h1>
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
    <link rel="canonical" href="{BASE_URL}/">
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


@app.get("/robots.txt", response_class=Response)
def robots_txt():
    content = f"""User-agent: *
Allow: /
Disallow: /v2/
Disallow: /summarize
Disallow: /docs
Disallow: /redoc
Disallow: /openapi.json

Sitemap: {BASE_URL}/sitemap.xml
"""

    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
    )


@app.get("/sitemap.xml", response_class=Response)
def sitemap_xml():
    public_paths = [
        "/",
        "/about",
        "/contact",
        "/download-app",
        "/privacy-policy",
        "/terms-and-conditions",
    ]
    today = "2026-07-03"
    urls = "\n".join(
        (
            "  <url>\n"
            f"    <loc>{html.escape(BASE_URL + path)}</loc>\n"
            f"    <lastmod>{today}</lastmod>\n"
            "    <changefreq>monthly</changefreq>\n"
            "    <priority>0.7</priority>\n"
            "  </url>"
        )
        for path in public_paths
    )
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
{urls}
</urlset>
"""

    return Response(
        content=content,
        media_type="application/xml; charset=utf-8",
    )


@app.get("/privacy")
def privacy_redirect():
    return RedirectResponse(
        url="/privacy-policy",
        status_code=301,
    )


@app.get("/terms")
def terms_redirect():
    return RedirectResponse(
        url="/terms-and-conditions",
        status_code=301,
    )


@app.get("/about", response_class=HTMLResponse)
def about_page():
    body = """
        <p>
            Lumina AI is an AI-powered document intelligence and notes
            summarization product for students, professionals, and general
            productivity workflows.
        </p>
        <p>
            The service helps turn text, PDFs, scanned pages, images, and camera
            OCR results into structured outputs such as summaries, reports,
            tables, meeting minutes, action items, flashcards, Q&A, email drafts,
            study notes, and simplified explanations.
        </p>
        <p>
            Public API and website routes are served from the Lumina AI domain,
            while authenticated document processing, usage limits, and saved
            workspace features are handled by the app and backend services.
        </p>
    """

    return legal_page(
        "About Lumina AI",
        "AI document intelligence for study, work, and everyday clarity.",
        body,
        description=(
            "Learn about Lumina AI, an AI-powered document intelligence and "
            "notes summarization product for PDFs, scans, images, and text."
        ),
        canonical_path="/about",
    )


@app.get("/contact", response_class=HTMLResponse)
def contact_page():
    body = f"""
        <p>
            For Lumina AI support, privacy requests, grievance requests,
            account questions, or legal-page feedback, contact the official
            support address below.
        </p>
        <p>
            Email: <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </p>
    """

    return legal_page(
        "Contact Lumina AI",
        "Support and privacy contact information.",
        body,
        description=(
            "Contact Lumina AI for support, privacy requests, account questions, "
            "and legal-page feedback."
        ),
        canonical_path="/contact",
    )


@app.get("/app-version")
def app_version():
    return JSONResponse(
        {
        "app": APP_NAME,
        "latestVersionName": APP_VERSION_NAME,
        "latestVersionCode": APP_VERSION_CODE,
        "minimumSupportedVersionCode": 1,
        "forceUpdate": False,
        "downloadUrl": app_download_url(),
        "updatePageUrl": f"{BASE_URL}/update",
        "releaseNotes": APP_RELEASE_NOTES,
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        },
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


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
    <link rel="canonical" href="{BASE_URL}/download-app">
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
                <a class="button" id="download-apk-button" href="{APP_DOWNLOAD_PATH}?v={APP_VERSION_CODE}" download>Download APK v{APP_VERSION_NAME}</a>
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
    <script>
        const downloadButton = document.getElementById("download-apk-button");
        if (downloadButton) {{
            let clicked = false;
            downloadButton.addEventListener("click", () => {{
                if (clicked) {{
                    return;
                }}
                clicked = true;
                downloadButton.textContent = "Downloading Lumina AI...";
                window.setTimeout(() => {{
                    clicked = false;
                    downloadButton.textContent = "Download APK v{APP_VERSION_NAME}";
                }}, 6000);
            }});
        }}
    </script>
</body>
</html>
"""


@app.get("/update", response_class=HTMLResponse)
def update_app():
    return download_app()


@app.get("/privacy-policy", response_class=HTMLResponse)
def privacy_policy():
    body = f"""
        <p><strong>Effective Date:</strong> July 3, 2026</p>
        <p><strong>Last Updated:</strong> July 3, 2026</p>

        <p>
            Lumina AI is an AI-powered document intelligence and notes
            summarization service. This Privacy Policy explains how information
            is collected, used, processed, and protected when users access the
            Lumina AI app, website, backend API, APK/update pages, and related
            services.
        </p>

        <div class="notice">
            This policy is intended to support clear privacy transparency,
            including under applicable Indian data protection principles and the
            Digital Personal Data Protection Act, 2023 where applicable. It does
            not claim that every possible legal requirement is fully satisfied in
            every jurisdiction.
        </div>

        <h2>Information We Collect</h2>
        <ul>
            <li>Account information such as name, username, email address, user ID, profile photo URL, sign-in provider, email verification status, account creation date, last login time, recovery email where provided, and profile updates.</li>
            <li>Authentication information handled by Firebase Authentication, including email/password login, Google sign-in, verification email status, password reset flows, and session tokens.</li>
            <li>User content such as pasted text, uploaded PDFs, uploaded images, camera scans, OCR text, extracted document text, selected AI mode, selected task, and AI-generated outputs.</li>
            <li>Saved workspace information such as summaries, markdown, plain text, sections, folders, favorites, pinned status, document history, search tokens, timestamps, reading estimates, and user-specific usage counters.</li>
            <li>Usage and routing information such as request counts, timestamps, task type, mode, processing route, model metadata, cache status, daily limit usage, feedback ratings, and processing diagnostics.</li>
            <li>Device, browser, app, and diagnostic information such as app opens, crashes, errors, performance events, update prompts, and feature usage where Firebase Analytics or Crashlytics are enabled.</li>
            <li>Technical security information such as IP-derived rate-limit data, request timing, API errors, service health information, and abuse-prevention signals.</li>
            <li>Cookies, local storage, or similar local device storage may be used by the Flutter web/app runtime, Firebase, authentication flows, and local app preferences such as onboarding and appearance settings.</li>
        </ul>

        <h2>How We Use Information</h2>
        <ul>
            <li>To authenticate users and protect account access.</li>
            <li>To extract text from images, camera scans, and PDFs using app-side and backend-side OCR/extraction features.</li>
            <li>To process submitted text and extracted content with AI models and return summaries, tables, reports, notes, action items, flashcards, Q&A, email drafts, and other requested outputs.</li>
            <li>To save generated documents, folders, favorites, profile settings, feedback, and usage limits to the user's account where those features are used.</li>
            <li>To detect crashes, measure reliability, prevent abuse, apply rate limits, and improve the product experience.</li>
            <li>To cache or reuse responses where configured, improve reliability, debug errors, operate job queues, and reduce repeated processing.</li>
            <li>To communicate important service or update information when applicable.</li>
        </ul>

        <h2>Uploaded Documents and AI Processing</h2>
        <p>
            Users may upload or paste documents for processing. Uploaded content
            and extracted text are processed to generate the requested output.
            AI-generated outputs may not always be accurate, complete, current,
            or suitable for a user's specific purpose. Users should review outputs
            before relying on them and should not upload documents they do not
            have the right to process. Users should avoid uploading highly
            sensitive information unless it is necessary for their use of the
            service.
        </p>

        <h2>Third-Party Services</h2>
        <ul>
            <li>Firebase Authentication for sign-in, verification, password reset, and account identity.</li>
            <li>Cloud Firestore for user profiles, summaries, folders, favorites, and daily usage counters.</li>
            <li>Firebase App Check, Analytics, and Crashlytics for abuse protection, diagnostics, app-open analytics, and crash reports.</li>
            <li>Google sign-in where the user chooses Google authentication.</li>
            <li>Google Gemini or Google AI services for AI generation where configured by the backend.</li>
            <li>Google ML Kit or device OCR libraries for image and camera text recognition in the app where supported.</li>
            <li>Redis-compatible caching infrastructure where REDIS_URL is configured for exact response caching or queue support.</li>
            <li>Hugging Face Spaces and Cloudflare Worker/proxy infrastructure where used to host or route the backend and public website.</li>
        </ul>
        <p>
            These providers may process data as needed to deliver authentication,
            storage, analytics, crash reporting, hosting, OCR, caching, and AI
            generation features.
        </p>

        <h2>Legal Basis and Lawful Use</h2>
        <p>
            Lumina AI processes information to provide the service requested by
            users, operate accounts, generate AI outputs, maintain security,
            enforce limits, prevent abuse, and support service reliability. Users
            provide document content voluntarily when they paste, upload, scan, or
            save content in the service.
        </p>

        <h2>Data Retention</h2>
        <p>
            Uploaded files handled by the backend are written to temporary storage
            for processing and the backend code deletes those temporary files after
            processing completes. Extracted text, generated summaries, folders,
            favorites, profile records, usage counters, statistics, and saved
            workspace data may be retained while the account remains active or
            until deleted by the user where deletion features are available.
            Cached/generated outputs may be stored temporarily where caching is
            configured. Logs, diagnostic records, security records, and provider
            records may be retained for a limited period where needed for service
            operation, debugging, legal obligations, abuse prevention, or limit
            enforcement.
        </p>

        <h2>Data Sharing</h2>
        <p>
            Lumina AI does not sell user data. Information may be shared with
            service providers that help operate authentication, storage, hosting,
            analytics, crash reporting, OCR, caching, and AI generation. Data may
            also be disclosed where required by law, to protect users or the
            service, to investigate abuse or security issues, or with user consent.
        </p>

        <h2>Security</h2>
        <p>
            Lumina AI uses reasonable technical and organizational safeguards for
            the current service, including HTTPS for the public domain, Firebase
            authentication, user-specific records, backend rate limits, App Check
            where enabled, trusted host checks, upload size limits, and security
            headers. No online system can be guaranteed completely secure.
        </p>

        <h2>User Rights and Choices</h2>
        <ul>
            <li>Users may request access to information associated with their account.</li>
            <li>Users may correct profile information supported by the app.</li>
            <li>Users may delete generated summaries and folders from inside the app.</li>
            <li>Users may reset passwords, update profile information, and manage sign-in methods supported by Firebase.</li>
            <li>Users may delete their account and associated saved data from the Profile section.</li>
            <li>Users may withdraw consent where applicable by stopping use of the service or requesting deletion, subject to data needed for security, legal, or operational reasons.</li>
            <li>Users may contact support for privacy, deletion, grievance, or user-rights assistance.</li>
        </ul>

        <h2>Children's Privacy</h2>
        <p>
            Lumina AI is not intended for children below the legally relevant age
            unless parental, guardian, or school consent is provided where required.
            Lumina AI does not knowingly collect children's personal data without
            appropriate consent.
        </p>

        <h2>International Processing</h2>
        <p>
            Because Lumina AI uses providers such as Google/Firebase, Google AI
            services, Hugging Face, Cloudflare, and Redis-compatible infrastructure
            where configured, data may be processed in countries other than the
            user's country depending on provider infrastructure.
        </p>

        <h2>AI Output Disclaimer</h2>
        <p>
            Outputs are generated by AI and may contain mistakes, omissions,
            formatting errors, or misleading interpretations. Users are
            responsible for verifying outputs before academic, professional,
            legal, medical, financial, or other important use.
        </p>

        <h2>Changes to This Policy</h2>
        <p>
            This Privacy Policy may be updated as Lumina AI changes. The Last
            Updated date will be changed when material updates are made.
        </p>

        <h2>Contact and Grievance Requests</h2>
        <p>
            For privacy questions, grievance requests, user-rights requests, or
            deletion assistance, contact:
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </p>
    """

    return legal_page(
        "Privacy Policy | Lumina AI",
        "Learn how Lumina AI collects, uses, protects, and processes information.",
        body,
        description=(
            "Learn how Lumina AI collects, uses, protects, and processes "
            "information when users access AI-powered document summarization "
            "and document intelligence tools."
        ),
        canonical_path="/privacy-policy",
        heading="Privacy Policy",
    )


@app.get("/terms-and-conditions", response_class=HTMLResponse)
def terms_and_conditions():
    body = f"""
        <p><strong>Effective Date:</strong> July 3, 2026</p>
        <p><strong>Last Updated:</strong> July 3, 2026</p>

        <p>
            By accessing or using Lumina AI, you agree to these Terms and
            Conditions. If you do not agree, please do not use the service.
        </p>

        <h2>Description of Service</h2>
        <p>
            Lumina AI provides AI-powered document summarization, OCR-assisted
            extraction, document cleanup, and structured output tools. Depending
            on the selected mode and task, the service may generate summaries,
            tables, professional reports, meeting minutes, action items, email
            drafts, study notes, flashcards, Q&A, revision sheets, simplified
            explanations, and related document intelligence outputs.
        </p>

        <h2>Eligibility</h2>
        <p>
            Users must be legally able to use the service. Minors should use
            Lumina AI only with parent, guardian, or school consent where required
            by applicable law.
        </p>

        <h2>User Accounts</h2>
        <p>
            Lumina AI supports account-based features through Firebase
            Authentication and related services. Users are responsible for
            providing accurate account information, keeping credentials secure,
            maintaining control of their devices, and reporting suspected misuse
            of their account.
        </p>

        <h2>User Content</h2>
        <p>
            Users retain rights they already have in documents, text, images,
            scans, and other content they upload, paste, or save. Users grant
            Lumina AI a limited permission to process that content only as needed
            to provide the requested service, including extraction, OCR,
            generation, saving, caching where configured, debugging, security,
            and usage-limit enforcement.
        </p>
        <ul>
            <li>Users must have the right to upload, process, summarize, or store the documents they submit.</li>
            <li>Users must not upload illegal, harmful, abusive, infringing, privacy-invasive, malicious, or unauthorized third-party content.</li>
            <li>Users must not upload malware or content designed to disrupt Lumina AI or another user's device, account, or data.</li>
            <li>Users are responsible for reviewing whether confidential, personal, regulated, academic, or workplace documents may be processed through the service.</li>
        </ul>

        <h2>AI Outputs</h2>
        <p>
            Outputs are generated automatically by AI systems and may be
            inaccurate, incomplete, outdated, misformatted, or unsuitable for a
            particular purpose. Users must verify important outputs before relying
            on them. Lumina AI is not a substitute for legal, medical, financial,
            academic-integrity, or other professional advice.
        </p>

        <h2>Acceptable Use</h2>
        <ul>
            <li>Do not use Lumina AI for illegal, harmful, deceptive, abusive, or infringing activity.</li>
            <li>Do not violate another person's privacy, intellectual property, confidentiality, or data rights.</li>
            <li>Do not attempt to access another user's account, documents, saved data, jobs, or usage records.</li>
            <li>Do not bypass rate limits, daily limits, file limits, authentication, App Check, security controls, or abuse-prevention systems.</li>
            <li>Do not scrape, overload, spam, probe, reverse engineer protected service behavior, or automate abusive requests against the app, API, or website.</li>
            <li>Do not upload malware, exploit payloads, or content intended to damage systems or interfere with service operation.</li>
        </ul>

        <h2>Plans, Limits, and Availability</h2>
        <p>
            Lumina AI may offer free or limited-access features. The backend code
            includes daily generation limits, rate limits, upload size limits,
            endpoint limits, PDF page checks, and heavy-processing safeguards.
            Current default backend settings include a 15 MB upload limit, a daily
            free generation limit configured by the service, and OCR/PDF limits
            that may vary by deployment. The service may be changed, paused,
            limited, or made unavailable for maintenance, safety, abuse
            prevention, cost control, provider outages, quota limits, or technical
            reasons.
        </p>

        <h2>Payments and Refunds</h2>
        <p>
            Currently, Lumina AI may offer free or limited-access features. Paid
            plans, if introduced later, will be governed by additional payment
            terms. No pricing, refund, subscription, or billing promise is made in
            these Terms unless a separate paid-plan policy is published.
        </p>

        <h2>Intellectual Property</h2>
        <p>
            Lumina AI branding, website content, app interface, backend code,
            design elements, and service features are owned by their respective
            rights holders. User content remains the user's responsibility. These
            Terms do not give Lumina AI ownership of user documents beyond the
            limited processing permission described above.
        </p>

        <h2>Third-Party Services</h2>
        <p>
            Lumina AI relies on third-party services for parts of the product,
            including Firebase authentication and storage, Google sign-in, Google
            Gemini or Google AI services, Google/Firebase analytics and crash
            diagnostics, OCR-related libraries or services, Redis-compatible
            caching where configured, Hugging Face hosting, and Cloudflare proxy
            infrastructure. Those services may be subject to their own terms,
            policies, availability, and technical limits.
        </p>

        <h2>Privacy</h2>
        <p>
            Use of Lumina AI is also governed by the
            <a href="/privacy-policy">Privacy Policy</a>.
        </p>

        <h2>Suspension or Termination</h2>
        <p>
            Access may be suspended, limited, or terminated if a user violates
            these Terms, creates a security risk, attempts abuse, exceeds fair-use
            boundaries, interferes with the service, or where action is required
            for legal, safety, provider, or operational reasons.
        </p>

        <h2>Disclaimers</h2>
        <p>
            Lumina AI is provided on an "as is" and "as available" basis. Lumina
            AI does not guarantee uninterrupted service, error-free operation,
            indexing by search engines, perfect OCR, complete extraction,
            accurate AI output, or suitability for any specific academic,
            professional, legal, medical, financial, or business purpose.
        </p>

        <h2>Limitation of Liability</h2>
        <p>
            To the maximum extent permitted by applicable law, Lumina AI and its
            operators are not liable for indirect, incidental, consequential,
            special, punitive, or similar losses, or for losses arising from user
            content, AI output errors, third-party services, service downtime,
            account misuse, unsupported devices, or reliance on generated content.
            Nothing in these Terms limits liability where doing so is not allowed
            by applicable law.
        </p>

        <h2>Indemnity</h2>
        <p>
            Users agree to be responsible for claims, losses, liabilities, damages,
            costs, and expenses arising from their misuse of Lumina AI, unlawful
            content, infringement of third-party rights, violation of privacy or
            confidentiality obligations, or violation of these Terms.
        </p>

        <h2>Changes to Terms</h2>
        <p>
            These Terms may be updated as Lumina AI changes. The Last Updated date
            will be changed when material updates are made. Continued use of the
            service after updates means the user accepts the updated Terms.
        </p>

        <h2>Governing Law</h2>
        <p>
            These Terms are intended to be governed by the laws of India, unless
            another jurisdiction is required by applicable law.
        </p>

        <h2>Contact</h2>
        <p>
            For support, legal, terms, privacy, or grievance questions, contact:
            <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>
        </p>
    """

    return legal_page(
        "Terms and Conditions | Lumina AI",
        "Read the terms that govern use of Lumina AI.",
        body,
        description=(
            "Read the terms that govern use of Lumina AI, including user "
            "content, AI-generated outputs, acceptable use, service limits, "
            "and disclaimers."
        ),
        canonical_path="/terms-and-conditions",
        heading="Terms and Conditions",
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
    weight: int = 1,
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

            if current_count + weight > DAILY_FREE_LIMIT:
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
                    "count": current_count + weight,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                },
            )

            return current_count + weight

        transaction.set(
            usage_ref,
            {
                "uid": user_uid,
                "date": today,
                "count": weight,
                "limit": DAILY_FREE_LIMIT,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

        return weight

    return update_usage(
        transaction,
        usage_ref,
    )


def get_daily_usage_count(user_uid: str) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage_ref = firestore_db.collection("usage").document(
        f"{user_uid}_{today}"
    )
    snapshot = usage_ref.get()

    if not snapshot.exists:
        return 0

    data = snapshot.to_dict() or {}
    return int(data.get("count", 0) or 0)


@app.post("/v2/generate")
@limiter.limit("20/minute")
def generate_v2(
    request: Request,
    generate_request: GenerateRequest,
    authorization: str = Header(None),
):
    print(
        "V2 generation request received:",
        {
            "client": request.headers.get("x-lumina-client", ""),
            "mode": generate_request.mode,
            "task": generate_request.task,
            "input_chars": len(generate_request.text or ""),
        },
    )

    decoded_user = verify_firebase_user(
        authorization,
    )

    user_uid = decoded_user.get("uid")

    if not user_uid:
        raise HTTPException(
            status_code=401,
            detail="Invalid Firebase user",
    )
    client_ip = safety_controls.client_ip(request)
    enforce_endpoint_safety(
        user_uid=user_uid,
        client_ip=client_ip,
        endpoint="v2_generate",
    )

    print("V2 generation auth passed:", {"uid": user_uid})

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
    started_at = time.perf_counter()
    request_id = anonymized_request_id(
        user_uid=user_uid,
        selected_mode=selected_mode,
        selected_task=selected_task,
        cleaned_text=cleaned_text,
    )
    route_metadata: Dict[str, Any] = {}

    try:
        print(
            "V2 generation pipeline started:",
            {
                "uid": user_uid,
                "mode": selected_mode,
                "task": selected_task,
                "input_chars": len(cleaned_text),
            },
        )

        prepared = lumina_router.process_text_for_mode(
            text=cleaned_text,
            mode=selected_mode,
            task=selected_task,
        )
        route_metadata = extract_route_metadata(prepared)
        enforce_heavy_safety(
            user_uid=user_uid,
            client_ip=client_ip,
            route_metadata=route_metadata,
            allow_queue=False,
        )
        cache_key = request_hasher.cache_key(
            user_id=user_uid,
            task=selected_task,
            mode=selected_mode,
            text=cleaned_text,
        )

        cached_payload = exact_cache.get(cache_key)

        if cached_payload_matches_request(
            cached_payload,
            selected_mode=selected_mode,
            selected_task=selected_task,
        ):
            cached_payload["cacheHit"] = True
            cached_payload["cached"] = True
            cached_payload["cacheType"] = "exact"
            cached_payload["usageCount"] = get_daily_usage_count(user_uid)
            cached_payload["dailyLimit"] = DAILY_FREE_LIMIT
            cached_payload.update(route_metadata)
            cached_payload = schema_validator.validate_generation_response(
                cached_payload,
            )

            print(
                "V2 generation cache hit:",
                {
                    "uid": user_uid,
                    "mode": selected_mode,
                    "task": selected_task,
                    "cache_type": "exact",
                },
            )
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=True,
            )

            return JSONResponse(content=cached_payload)

        similar_payload = semantic_cache.find_similar_request(
            text=cleaned_text,
            task=selected_task,
            mode=selected_mode,
        )

        if similar_payload:
            similar_payload["cacheHit"] = True
            similar_payload["cached"] = True
            similar_payload["cacheType"] = "semantic"
            similar_payload["usageCount"] = get_daily_usage_count(user_uid)
            similar_payload["dailyLimit"] = DAILY_FREE_LIMIT
            similar_payload.update(route_metadata)
            similar_payload = schema_validator.validate_generation_response(
                similar_payload,
            )
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=True,
            )

            print(
                "V2 generation cache hit:",
                {
                    "uid": user_uid,
                    "mode": selected_mode,
                    "task": selected_task,
                    "cache_type": "semantic",
                },
            )
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=True,
            )

            return JSONResponse(content=similar_payload)

        result = lumina_router.generate_prepared(
            prepared=prepared,
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
                status_code=generation_error_status(result),
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
            weight=usage_weight_for_route(route_metadata),
        )

        print(
            "V2 generation completed:",
            {
                "uid": user_uid,
                "mode": selected_mode,
                "task": selected_task,
                "output_chars": len(generated_text),
            },
        )

        response_payload = build_generate_response_payload(
            formatted=formatted,
            selected_mode=selected_mode,
            selected_task=selected_task,
            usage_count=usage_count,
            route_metadata=extract_route_metadata(result),
        )

        exact_cache.set(cache_key, response_payload)
        semantic_cache.save_embedding_cache(
            text=cleaned_text,
            task=selected_task,
            mode=selected_mode,
            response=response_payload,
            metadata=route_metadata,
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=extract_route_metadata(result),
            started_at=started_at,
            cache_hit=False,
        )

        return JSONResponse(
            content=response_payload
        )

    except HTTPException:
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type="http_exception",
        )
        raise

    except SafetyError as e:
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=e.error_type,
        )
        raise_safety_http_error(e)

    except Exception as e:
        print("V2 generation error:", str(e))
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=e.__class__.__name__,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to generate output. Please try again.",
        )


@app.post("/v2/jobs/generate")
@limiter.limit("20/minute")
def create_generation_job(
    request: Request,
    background_tasks: BackgroundTasks,
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
    client_ip = safety_controls.client_ip(request)
    enforce_endpoint_safety(
        user_uid=user_uid,
        client_ip=client_ip,
        endpoint="v2_jobs_generate",
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
    started_at = time.perf_counter()
    request_id = anonymized_request_id(
        user_uid=user_uid,
        selected_mode=selected_mode,
        selected_task=selected_task,
        cleaned_text=cleaned_text,
    )
    route_metadata: Dict[str, Any] = {}

    try:
        prepared = lumina_router.process_text_for_mode(
            text=cleaned_text,
            mode=selected_mode,
            task=selected_task,
        )
        route_metadata = extract_route_metadata(prepared)
        route_config = route_metadata.get("routeConfig", {})
        enforce_heavy_safety(
            user_uid=user_uid,
            client_ip=client_ip,
            route_metadata=route_metadata,
            allow_queue=True,
        )
        cache_key = request_hasher.cache_key(
            user_id=user_uid,
            task=selected_task,
            mode=selected_mode,
            text=cleaned_text,
        )

        cached_payload = exact_cache.get(cache_key)

        if cached_payload_matches_request(
            cached_payload,
            selected_mode=selected_mode,
            selected_task=selected_task,
        ):
            cached_payload["cacheHit"] = True
            cached_payload["cached"] = True
            cached_payload["cacheType"] = "exact"
            cached_payload["usageCount"] = get_daily_usage_count(user_uid)
            cached_payload["dailyLimit"] = DAILY_FREE_LIMIT
            cached_payload.update(route_metadata)
            cached_payload = schema_validator.validate_generation_response(
                cached_payload,
            )

            return JSONResponse(
                content={
                    "success": True,
                    "queued": False,
                    "status": "completed",
                    "cacheHit": True,
                    "cacheType": "exact",
                    "result": cached_payload,
                    **route_metadata,
                }
            )

        similar_payload = semantic_cache.find_similar_request(
            text=cleaned_text,
            task=selected_task,
            mode=selected_mode,
        )

        if similar_payload:
            similar_payload["cacheHit"] = True
            similar_payload["cached"] = True
            similar_payload["cacheType"] = "semantic"
            similar_payload["usageCount"] = get_daily_usage_count(user_uid)
            similar_payload["dailyLimit"] = DAILY_FREE_LIMIT
            similar_payload.update(route_metadata)
            similar_payload = schema_validator.validate_generation_response(
                similar_payload,
            )

            return JSONResponse(
                content={
                    "success": True,
                    "queued": False,
                    "status": "completed",
                    "cacheHit": True,
                    "cacheType": "semantic",
                    "result": similar_payload,
                    **route_metadata,
                }
            )

        if route_config.get("queue_required") is True:
            try:
                job = job_status_store.create_job(
                    user_id=user_uid,
                    route_metadata=route_metadata,
                )
                queue_mode = task_queue.enqueue(
                    background_tasks=background_tasks,
                    job_id=job["jobId"],
                    handler=process_generation_job,
                    payload={
                        "user_uid": user_uid,
                        "cleaned_text": cleaned_text,
                        "selected_mode": selected_mode,
                        "selected_task": selected_task,
                        "cache_key": cache_key,
                        "request_id": request_id,
                    },
                )

                return JSONResponse(
                    status_code=202,
                    content={
                        "success": True,
                        "queued": True,
                        "jobId": job["jobId"],
                        "status": "pending",
                        "statusUrl": f"/v2/jobs/{job['jobId']}",
                        "queueMode": queue_mode,
                        **route_metadata,
                    },
                )
            except Exception as exc:
                print("Background job unavailable:", exc.__class__.__name__)
                response_payload = generate_text_response_payload(
                    user_uid=user_uid,
                    cleaned_text=cleaned_text,
                    selected_mode=selected_mode,
                    selected_task=selected_task,
                    prepared=prepared,
                )
                exact_cache.set(cache_key, response_payload)
                semantic_cache.save_embedding_cache(
                    text=cleaned_text,
                    task=selected_task,
                    mode=selected_mode,
                    response=response_payload,
                    metadata=route_metadata,
                )
                log_routing_event(
                    user_uid=user_uid,
                    request_id=request_id,
                    route_metadata=route_metadata,
                    started_at=started_at,
                    cache_hit=False,
                    error_type="job_fallback_inline",
                )

                return JSONResponse(
                    content={
                        "success": True,
                        "queued": False,
                        "status": "completed",
                        "queueMode": "inline_fallback",
                        "result": response_payload,
                        **route_metadata,
                    }
                )

        response_payload = generate_text_response_payload(
            user_uid=user_uid,
            cleaned_text=cleaned_text,
            selected_mode=selected_mode,
            selected_task=selected_task,
            prepared=prepared,
        )
        exact_cache.set(cache_key, response_payload)
        semantic_cache.save_embedding_cache(
            text=cleaned_text,
            task=selected_task,
            mode=selected_mode,
            response=response_payload,
            metadata=route_metadata,
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
        )

        return JSONResponse(
            content={
                "success": True,
                "queued": False,
                "status": "completed",
                "result": response_payload,
                **route_metadata,
            }
        )
    except HTTPException:
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type="http_exception",
        )
        raise
    except SafetyError as exc:
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=exc.error_type,
        )
        raise_safety_http_error(exc)

    except Exception as exc:
        print("V2 job creation error:", exc.__class__.__name__)
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
            error_type=exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create generation job. Please try again.",
        )


@app.get("/v2/jobs/{job_id}")
@limiter.limit("60/minute")
def get_generation_job(
    request: Request,
    job_id: str,
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

    job = job_status_store.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found",
        )

    if job.get("userId") != user_uid:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this job",
        )

    return JSONResponse(
        content={
            "success": True,
            "jobId": job.get("jobId"),
            "status": job.get("status"),
            "result": job.get("result"),
            "error": job.get("error", ""),
            "routeConfig": job.get("routeConfig", {}),
            "complexity": job.get("complexity", {}),
            "createdAt": job.get("createdAt"),
            "updatedAt": job.get("updatedAt"),
        }
    )


@app.post("/v2/feedback")
@limiter.limit("30/minute")
def submit_generation_feedback(
    request: Request,
    feedback_request: FeedbackRequest,
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

    client_ip = safety_controls.client_ip(request)
    enforce_endpoint_safety(
        user_uid=user_uid,
        client_ip=client_ip,
        endpoint="v2_feedback",
    )

    allowed_tags = {
        "too short",
        "too long",
        "missing points",
        "bad formatting",
        "OCR mistake",
        "very helpful",
    }
    clean_tags = [
        tag
        for tag in feedback_request.tags[:8]
        if tag in allowed_tags
    ]

    try:
        firestore_db.collection("feedback").add(
            {
                "uid": user_uid,
                "title": feedback_request.title.strip()[:160],
                "mode": validate_mode(feedback_request.mode),
                "task": feedback_request.task.strip()[:50],
                "rating": feedback_request.rating,
                "tags": clean_tags,
                "route": feedback_request.route.strip()[:40],
                "modelTier": feedback_request.modelTier.strip()[:40],
                "source": "flutter_generation_feedback",
                "createdAt": firestore.SERVER_TIMESTAMP,
            }
        )
    except Exception as exc:
        print("Feedback save error:", str(exc))
        raise HTTPException(
            status_code=500,
            detail="Failed to save feedback. Please try again.",
        )

    return JSONResponse(
        content={
            "success": True,
        }
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

    client_ip = safety_controls.client_ip(request)
    enforce_endpoint_safety(
        user_uid=user_uid,
        client_ip=client_ip,
        endpoint="v2_generate_file",
    )

    selected_mode = validate_mode(mode)
    selected_task = validate_task(selected_mode, task)
    started_at = time.perf_counter()
    request_id = ""
    route_metadata: Dict[str, Any] = {}

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

        request_id = anonymized_file_request_id(temp_path)

        if extension == ".pdf":
            try:
                safety_controls.validate_pdf_pages(
                    temp_path,
                    user_plan="free",
                )
            except SafetyError as error:
                safety_controls.register_violation(
                    user_id=user_uid,
                    ip=client_ip,
                )
                raise_safety_http_error(error)

        result = safety_controls.run_with_timeout(
            lambda: lumina_router.generate_from_file(
                file_path=temp_path,
                mode=selected_mode,
                task=selected_task,
            ),
            timeout_seconds=safety_controls.ocr_timeout_seconds,
        )
        route_metadata = extract_route_metadata(result)
        enforce_heavy_safety(
            user_uid=user_uid,
            client_ip=client_ip,
            route_metadata=route_metadata,
            allow_queue=True,
        )

        formatted = result.get("formatted_output", {})
        generated_text = formatted.get("markdown", "")

        if not generated_text:
            error_detail = extract_generation_error(result)

            raise HTTPException(
                status_code=generation_error_status(result),
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
            weight=usage_weight_for_route(route_metadata),
        )
        pipeline_output = result.get("pipeline_output", {})
        extraction = pipeline_output.get("extraction", {})
        structure = pipeline_output.get("structure", {})
        response_payload = build_generate_response_payload(
            formatted=formatted,
            selected_mode=selected_mode,
            selected_task=selected_task,
            usage_count=usage_count,
            cache_hit=False,
            cache_type="",
            route_metadata=route_metadata,
        )
        response_payload.update(
            {
                "extractionSource": extraction.get("source", ""),
                "extractionConfidence": extraction.get("confidence", 0),
                "extractionQuality": extraction.get("quality", {}),
                "extractionPreview": extraction.get("quality", {}).get(
                    "preview",
                    "",
                ),
                "tableCount": structure.get("metadata", {}).get(
                    "table_count",
                    0,
                ),
                "parserType": structure.get("parser_type", ""),
            }
        )
        log_routing_event(
            user_uid=user_uid,
            request_id=request_id,
            route_metadata=route_metadata,
            started_at=started_at,
            cache_hit=False,
        )

        return JSONResponse(
            content=response_payload
        )

    except HTTPException:
        if request_id:
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=False,
                error_type="http_exception",
            )
        raise

    except SafetyError as e:
        if request_id:
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=False,
                error_type=e.error_type,
            )
        raise_safety_http_error(e)

    except Exception as e:
        print("V2 file generation error:", e.__class__.__name__)
        if request_id:
            log_routing_event(
                user_uid=user_uid,
                request_id=request_id,
                route_metadata=route_metadata,
                started_at=started_at,
                cache_hit=False,
                error_type=e.__class__.__name__,
            )

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

    client_ip = safety_controls.client_ip(request)
    enforce_endpoint_safety(
        user_uid=user_uid,
        client_ip=client_ip,
        endpoint="legacy_summarize",
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
        route_metadata = extract_route_metadata(result)
        enforce_heavy_safety(
            user_uid=user_uid,
            client_ip=client_ip,
            route_metadata=route_metadata,
            allow_queue=False,
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
                status_code=generation_error_status(result),
                detail=error_detail,
            )

        usage_count = check_and_increment_daily_usage(
            user_uid,
            weight=usage_weight_for_route(route_metadata),
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
        print("Legacy summarize via V2 pipeline error:", e.__class__.__name__)

        raise HTTPException(
            status_code=500,
            detail="Failed to generate summary. Please try again.",
        )

