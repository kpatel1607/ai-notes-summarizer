from typing import Any, Dict
from pathlib import Path
import os
import requests
from dotenv import load_dotenv
from requests import Timeout


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    BASE_DIR / ".env"
)


class GenerationService:
    """
    Temporary Gemini-only generation service.

    Ollama routing and fallback are intentionally disabled for now so every
    generation request goes directly to Gemini.
    """

    def __init__(self) -> None:
        # Force Gemini temporarily, regardless of any old environment value.
        self.provider = "gemini"

        self.gemini_model_name = os.getenv(
            "LUMINA_MODEL_NAME",
            "gemini-2.5-flash",
        ).strip()

        default_gemini_api_url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{self.gemini_model_name}:generateContent"
        )

        self.gemini_api_url = os.getenv(
            "LUMINA_API_URL",
            default_gemini_api_url,
        ).strip()

        self.gemini_api_key = os.getenv(
            "LUMINA_API_KEY",
            "",
        ).strip()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        """
        Send every request directly to Gemini.

        Ollama is temporarily disabled and is not consulted even if
        LUMINA_GENERATION_PROVIDER=ollama exists in the environment.
        """

        print(
            "\n[Generation Provider]"
            "\nGemini-only mode enabled."
            "\nOllama is temporarily disabled.\n"
        )

        return self._generate_with_gemini(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def max_tokens_for_task(
            self,
            task: str,
            input_word_count: int = 0,
            strategy: str = "single_pass_generation",
    ) -> int:
        task = (task or "").strip().lower()
        strategy = (
                strategy or "single_pass_generation"
        ).strip().lower()

        # Base output budget according to the expected output shape.
        task_budgets = {
            # General tasks
            "short_summary": 1600,
            "bullet_summary": 2000,
            "key_points": 1800,
            "simplify": 2400,
            "clean_text": 4200,

            # Student tasks
            "important_notes": 4200,
            "qa_generation": 4200,
            "answer_questions": 5000,
            "flashcards": 3600,
            "mcqs": 4600,
            "beginner_explanation": 3600,
            "revision_sheet": 4200,

            # Professional tasks
            "executive_summary": 2600,
            "main_points": 2400,
            "action_items": 2600,
            "meeting_minutes": 3600,
            "structured_report": 4200,
            "table_format": 3400,
            "email_draft": 1800,
        }

        budget = task_budgets.get(
            task,
            2200,
        )

        # Tasks whose output can naturally be longer than the source.
        expansion_factors = {
            "answer_questions": 6.0,
            "qa_generation": 4.0,
            "mcqs": 5.0,
            "flashcards": 3.5,
            "important_notes": 3.0,
            "revision_sheet": 3.0,
            "beginner_explanation": 3.0,
            "structured_report": 2.5,
            "meeting_minutes": 2.2,
            "table_format": 2.0,
            "action_items": 1.8,
            "main_points": 1.5,
            "executive_summary": 1.4,
            "bullet_summary": 1.5,
            "key_points": 1.5,
            "simplify": 1.5,
            "clean_text": 1.8,
            "short_summary": 0.9,
            "email_draft": 1.2,
        }

        expansion_factor = expansion_factors.get(
            task,
            1.5,
        )

        # Approximate English conversion:
        # one word is commonly around 1.3–1.7 tokens.
        estimated_task_tokens = int(
            max(input_word_count, 1)
            * expansion_factor
            * 1.6
        )

        budget = max(
            budget,
            estimated_task_tokens,
        )

        # Larger source documents need extra room even for concise tasks.
        if input_word_count > 500:
            budget = max(
                budget,
                2800,
            )

        if input_word_count > 1200:
            budget = max(
                budget,
                4200,
            )

        if input_word_count > 2500:
            budget = max(
                budget,
                6000,
            )

        # Final synthesis from multiple chunk outputs needs more room.
        if strategy == "hierarchical_summary":
            budget = max(
                budget,
                5000,
            )

        # Keep a safe application-level ceiling.
        # This is below Gemini's model maximum but large enough for your tasks.
        return min(
            budget,
            8000,
        )

    def _generate_with_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        if not self.gemini_api_url:
            return {
                "success": False,
                "error_type": "configuration_error",
                "error": "Gemini API URL not configured.",
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        if not self.gemini_api_key:
            return {
                "success": False,
                "error_type": "configuration_error",
                "error": "Gemini API key not configured.",
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        try:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "thinkingConfig": {
                        "thinkingBudget": 0,
                    },
                },
            }

            response = requests.post(
                self._build_gemini_url(),
                json=payload,
                headers={
                    "Content-Type": "application/json",
                },
                timeout=120,
            )

            if response.status_code == 429:
                return {
                    "success": False,
                    "error_type": "quota_exceeded",
                    "error": (
                        "Gemini quota exceeded. Please wait or check "
                        "your API quota."
                    ),
                    "generated_text": "",
                    "provider": "gemini",
                    "model": self.gemini_model_name,
                }

            if response.status_code >= 400:
                return {
                    "success": False,
                    "error_type": "provider_error",
                    "error": (
                        f"Gemini API error {response.status_code}: "
                        f"{response.text[:500]}"
                    ),
                    "generated_text": "",
                    "provider": "gemini",
                    "model": self.gemini_model_name,
                }

            data = response.json()

            candidates = data.get(
                "candidates",
                [],
            )

            candidate = (
                candidates[0]
                if candidates
                else {}
            )

            finish_reason = candidate.get(
                "finishReason",
                "",
            )

            usage_metadata = data.get(
                "usageMetadata",
                {},
            )

            generated_text = self._extract_gemini_text(
                data,
            )

            if finish_reason == "MAX_TOKENS":
                return {
                    "success": False,
                    "error_type": "max_tokens_reached",
                    "error": (
                        "Gemini reached the output-token limit "
                        "before completing the response."
                    ),
                    "generated_text": generated_text,
                    "raw_response": data,
                    "provider": "gemini",
                    "model": self.gemini_model_name,
                    "finish_reason": finish_reason,
                    "usage_metadata": usage_metadata,
                }

            generated_text = self._extract_gemini_text(
                data,
            )

            if not generated_text:
                return {
                    "success": False,
                    "error_type": "empty_provider_response",
                    "error": (
                        "Gemini returned no text. Raw response: "
                        f"{str(data)[:500]}"
                    ),
                    "generated_text": "",
                    "raw_response": data,
                    "provider": "gemini",
                    "model": self.gemini_model_name,
                }

            return {
                "success": True,
                "generated_text": generated_text,
                "raw_response": data,
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        except Timeout:
            return {
                "success": False,
                "error_type": "timeout",
                "error": "Gemini request timed out.",
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        except requests.RequestException as exc:
            return {
                "success": False,
                "error_type": "network_error",
                "error": str(exc),
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        except Exception as exc:
            return {
                "success": False,
                "error_type": "unexpected_error",
                "error": str(exc),
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

    def _build_gemini_url(
        self,
    ) -> str:
        separator = (
            "&"
            if "?" in self.gemini_api_url
            else "?"
        )

        return (
            f"{self.gemini_api_url}"
            f"{separator}key={self.gemini_api_key}"
        )

    def _extract_gemini_text(
        self,
        response_json: Dict[str, Any],
    ) -> str:
        try:
            candidates = response_json.get(
                "candidates",
                [],
            )

            if not candidates:
                return ""

            content = candidates[0].get(
                "content",
                {},
            )

            parts = content.get(
                "parts",
                [],
            )

            texts = [
                str(part.get("text", "") or "").strip()
                for part in parts
                if isinstance(part, dict)
                and part.get("text")
            ]

            return "\n".join(
                text
                for text in texts
                if text
            ).strip()

        except Exception:
            return ""


if __name__ == "__main__":
    service = GenerationService()

    result = service.generate(
        prompt="Return exactly: Gemini connection successful.",
        temperature=0.0,
        max_tokens=50,
    )

    print(result)
