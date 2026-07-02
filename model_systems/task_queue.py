import os
from typing import Any, Callable, Optional

from fastapi import BackgroundTasks


try:
    from celery import Celery
except Exception:  # pragma: no cover - optional production dependency
    Celery = None


class TaskQueue:
    def __init__(self):
        self.backend = os.getenv("LUMINA_QUEUE_BACKEND", "in_process").strip().lower()
        self.celery_broker_url = os.getenv("CELERY_BROKER_URL", "").strip()
        self.celery_backend_url = os.getenv("CELERY_RESULT_BACKEND", "").strip()
        self.celery_task_name = os.getenv(
            "LUMINA_CELERY_GENERATE_TASK",
            "lumina.generate_job",
        )
        self._celery_app = None

        if self.backend == "celery" and Celery and self.celery_broker_url:
            try:
                self._celery_app = Celery(
                    "lumina",
                    broker=self.celery_broker_url,
                    backend=self.celery_backend_url or None,
                )
            except Exception as exc:
                print("Celery queue disabled:", exc.__class__.__name__)
                self._celery_app = None

    @property
    def mode(self) -> str:
        if self._celery_app:
            return "celery"
        return "in_process"

    def enqueue(
        self,
        *,
        background_tasks: BackgroundTasks,
        job_id: str,
        handler: Callable[..., Any],
        payload: dict,
    ) -> str:
        if self._celery_app:
            try:
                self._celery_app.send_task(
                    self.celery_task_name,
                    kwargs={
                        "job_id": job_id,
                        **payload,
                    },
                )
                return "celery"
            except Exception as exc:
                print("Celery enqueue failed, using in-process queue:", exc.__class__.__name__)

        background_tasks.add_task(
            handler,
            job_id=job_id,
            **payload,
        )
        return "in_process"
