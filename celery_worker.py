import os

from celery import Celery

from main import process_generation_job


broker_url = os.getenv("CELERY_BROKER_URL", "").strip()
result_backend = os.getenv("CELERY_RESULT_BACKEND", "").strip() or broker_url

celery_app = Celery(
    "lumina",
    broker=broker_url,
    backend=result_backend or None,
)


@celery_app.task(name="lumina.generate_job")
def generate_job(job_id: str, **payload):
    process_generation_job(
        job_id=job_id,
        **payload,
    )
