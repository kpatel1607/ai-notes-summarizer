import json
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


try:
    import redis
except Exception:  # pragma: no cover - optional production dependency
    redis = None


class JobStatusStore:
    STATUSES = {"pending", "processing", "completed", "failed"}

    def __init__(self, redis_url: Optional[str] = None, ttl_seconds: Optional[int] = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "").strip()
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("LUMINA_JOB_TTL_SECONDS", "86400")
        )
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._client = None

        if redis and self.redis_url:
            try:
                self._client = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=3,
                )
                self._client.ping()
            except Exception as exc:
                print("Redis job status disabled:", exc.__class__.__name__)
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def create_job(
        self,
        *,
        user_id: str,
        route_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = self._now()
        job = {
            "jobId": uuid.uuid4().hex,
            "userId": user_id,
            "status": "pending",
            "result": None,
            "error": "",
            "routeConfig": (route_metadata or {}).get("routeConfig", {}),
            "complexity": (route_metadata or {}).get("complexity", {}),
            "createdAt": now,
            "updatedAt": now,
        }
        self.save(job)
        return job

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if self._client:
            try:
                raw = self._client.get(self._key(job_id))
                if raw:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:
                print("Job status read failed:", exc.__class__.__name__)
                return None

        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def save(self, job: Dict[str, Any]) -> None:
        job["updatedAt"] = self._now()

        if self._client:
            try:
                self._client.setex(
                    self._key(job["jobId"]),
                    self.ttl_seconds,
                    json.dumps(job, ensure_ascii=False),
                )
                return
            except Exception as exc:
                print("Job status write failed:", exc.__class__.__name__)

        with self._lock:
            self._jobs[job["jobId"]] = dict(job)

    def update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: str = "",
        route_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        job = self.get(job_id)

        if not job:
            return None

        if status:
            if status not in self.STATUSES:
                raise ValueError(f"Invalid job status: {status}")
            job["status"] = status

        if result is not None:
            job["result"] = result

        if error:
            job["error"] = error

        if route_metadata:
            job["routeConfig"] = route_metadata.get("routeConfig", job.get("routeConfig", {}))
            job["complexity"] = route_metadata.get("complexity", job.get("complexity", {}))

        self.save(job)
        return job

    def _key(self, job_id: str) -> str:
        return f"lumina:job:v1:{job_id}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()
