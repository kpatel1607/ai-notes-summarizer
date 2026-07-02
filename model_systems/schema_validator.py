from typing import Any, Dict, List


class SchemaValidator:
    REQUIRED_KEYS = {
        "title": "",
        "sections": [],
        "markdown": "",
        "plainText": "",
        "mode": "",
        "task": "",
        "route": "",
        "modelTier": "",
        "cached": False,
    }

    def validate_generation_response(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        validated = dict(payload or {})

        for key, default in self.REQUIRED_KEYS.items():
            if key not in validated or validated[key] is None:
                validated[key] = default

        validated["title"] = str(validated.get("title") or "Generated Document")
        validated["markdown"] = str(validated.get("markdown") or "")
        validated["plainText"] = str(
            validated.get("plainText")
            or validated.get("plain_text")
            or self._markdown_to_plain_text(validated["markdown"])
        )
        validated["sections"] = self._normalize_sections(
            validated.get("sections")
        )
        validated["mode"] = str(validated.get("mode") or "")
        validated["task"] = str(validated.get("task") or "")
        validated["route"] = str(validated.get("route") or "")
        validated["modelTier"] = str(validated.get("modelTier") or "")
        validated["cached"] = bool(validated.get("cached"))

        if "sectionCount" not in validated:
            validated["sectionCount"] = len(validated["sections"])

        if "cacheHit" not in validated:
            validated["cacheHit"] = validated["cached"]

        return validated

    def _normalize_sections(self, sections: Any) -> List[Dict[str, Any]]:
        if not isinstance(sections, list):
            return []

        normalized = []

        for section in sections:
            if not isinstance(section, dict):
                continue

            normalized.append({
                "heading": str(section.get("heading") or section.get("title") or ""),
                "level": int(section.get("level") or 2),
                "content": section.get("content") if isinstance(section.get("content"), list) else [],
            })

        return normalized

    def _markdown_to_plain_text(self, markdown: str) -> str:
        text = markdown.replace("#", "")
        text = text.replace("**", "")
        text = text.replace("*", "")
        return text.strip()
