import json
import os
from typing import Any, Dict, Optional


try:
    import redis
except Exception:  # pragma: no cover - optional production dependency
    redis = None


class RedisExactCache:
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "").strip()
        self.ttl_seconds = ttl_seconds or int(
            os.getenv("LUMINA_CACHE_TTL_SECONDS", "86400")
        )
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
                print("Redis cache disabled:", str(exc))
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._client:
            return None

        try:
            cached = self._client.get(key)

            if not cached:
                return None

            data = json.loads(cached)

            if isinstance(data, dict):
                return data
        except Exception as exc:
            print("Redis cache read failed:", str(exc))

        return None

    def set(self, key: str, value: Dict[str, Any]) -> bool:
        if not self._client:
            return False

        try:
            self._client.setex(
                key,
                self.ttl_seconds,
                json.dumps(value, ensure_ascii=False),
            )
            return True
        except Exception as exc:
            print("Redis cache write failed:", str(exc))
            return False
