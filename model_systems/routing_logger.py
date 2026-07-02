from datetime import datetime, timezone
from typing import Any, Dict, Optional


class RoutingLogger:
    def __init__(self, firestore_db=None):
        self.firestore_db = firestore_db

    def log(
        self,
        *,
        user_id: str,
        anonymized_request_id: str,
        route_metadata: Optional[Dict[str, Any]] = None,
        processing_time_ms: int = 0,
        cache_hit: bool = False,
        error_type: str = "",
        user_feedback_rating: Optional[int] = None,
    ) -> bool:
        if self.firestore_db is None:
            return False

        route_metadata = route_metadata or {}
        route_config = route_metadata.get("routeConfig", {})
        complexity = route_metadata.get("complexity", {})
        features = route_metadata.get("features", {})

        if not features:
            features = route_metadata.get("input_analysis", {}).get("metadata", {})

        log_payload = {
            "user_id": user_id,
            "anonymized_request_id": anonymized_request_id,
            "input_type": features.get("input_type")
            or route_metadata.get("input_analysis", {}).get("input_type", ""),
            "word_count": int(features.get("word_count") or 0),
            "page_count": int(features.get("page_count") or 0),
            "has_tables": bool(features.get("has_tables")),
            "has_images": bool(features.get("has_images")),
            "ocr_confidence": float(features.get("ocr_confidence") or 0.0),
            "layout_complexity": int(features.get("layout_complexity") or 0),
            "selected_route": route_config.get("path", ""),
            "selected_model_tier": route_config.get("model_tier", ""),
            "complexity_score": int(complexity.get("score") or 0),
            "complexity_reasons": complexity.get("reasons", []),
            "processing_time_ms": int(processing_time_ms),
            "cache_hit": bool(cache_hit),
            "error_type": error_type,
            "user_feedback_rating": user_feedback_rating,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            self.firestore_db.collection("routing_logs").add(log_payload)
            return True
        except Exception as exc:
            print("Routing log write failed:", str(exc))
            return False
