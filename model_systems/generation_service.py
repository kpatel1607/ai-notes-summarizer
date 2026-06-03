from typing import Dict, Any
from pathlib import Path
import re
import os
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(
    BASE_DIR / ".env"
)


class GenerationService:

    def __init__(self):

        self.provider = os.getenv(
            "LUMINA_GENERATION_PROVIDER",
            "gemini",
        ).lower().strip()

        self.gemini_model_name = os.getenv(
            "LUMINA_MODEL_NAME",
            "gemini-2.0-flash",
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

        self.ollama_url = os.getenv(
            "LUMINA_OLLAMA_URL",
            "http://localhost:11434/api/generate",
        )

        self.ollama_model = os.getenv(
            "LUMINA_OLLAMA_MODEL",
            "qwen2.5:1.5b",
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:

        if self.provider == "ollama":

            if self._should_use_gemini_directly(
                prompt,
            ):
                print(
                    "\n[Direct Gemini Routing]"
                    "\nLarge or complex prompt detected."
                    "\nSkipping Ollama.\n"
                )

                gemini_result = self._generate_with_gemini(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                gemini_result["direct_routing_used"] = True
                gemini_result["routing_reason"] = (
                    "large_or_complex_prompt"
                )

                return gemini_result

            ollama_result = self._generate_with_ollama(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            ollama_failed = (
                not ollama_result.get("success")
                or not ollama_result.get(
                    "generated_text",
                    "",
                ).strip()
            )

            ollama_low_quality = self._is_low_quality_output(
                generated_text=ollama_result.get(
                    "generated_text",
                    "",
                ),
                prompt=prompt,
            )

            if not ollama_failed and not ollama_low_quality:
                return ollama_result

            print(
                "\n[Fallback Activated]"
                "\nOllama generation failed or produced low-quality output."
                "\nSwitching to Gemini...\n"
            )

            gemini_result = self._generate_with_gemini(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            gemini_result["fallback_used"] = True
            gemini_result["fallback_from"] = self.ollama_model

            return gemini_result

        return self._generate_with_gemini(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _should_use_gemini_directly(
        self,
        prompt: str,
    ) -> bool:

        prompt_words = len(
            prompt.split()
        )

        # Qwen 1.5B is weak for large structured prompts.
        if prompt_words > 1800:
            return True

        lower_prompt = prompt.lower()

        complex_keywords = [
            "long document detected: true",
            "recommended strategy: hierarchical_summary",
            "free-tier limit applied: true",
            "only limited chunks were included",
            "document metadata:",
            "chunk summaries:",
            "multiple pages",
            "long document",
            "academic report",
            "research paper",
            "textbook",
            "internship report",
            "structured final answer",
        ]

        if any(
            keyword in lower_prompt
            for keyword in complex_keywords
        ):
            return True

        chunk_count = len(
            re.findall(
                r"\bchunk\s+\d+\b",
                lower_prompt,
            )
        )

        if chunk_count >= 6:
            return True

        return False

    def _is_low_quality_output(
        self,
        generated_text: str,
        prompt: str,
    ) -> bool:

        text = generated_text.strip()

        if not text:
            return True

        word_count = len(text.split())
        prompt_word_count = len(prompt.split())

        if prompt_word_count > 1200 and word_count < 250:
            return True

        unfinished_endings = [
            " or",
            " and",
            " with",
            " from",
            " to",
            " for",
            " of",
            " in",
            " the",
            " a",
            " an",
            "-",
            ":",
            ",",
        ]

        lower_text = text.lower()

        if any(
            lower_text.endswith(ending)
            for ending in unfinished_endings
        ):
            return True

        heading_count = (
            text.count("##")
            + text.count("# ")
        )

        if prompt_word_count > 1000 and heading_count < 4:
            return True

        if re.search(
            r"\n-\s*$",
            text,
        ):
            return True

        if re.search(
            r"\n-\s*[-–—]+\s*$",
            text,
        ):
            return True

        return False

    def _generate_with_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:

        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=300,
            )

            response.raise_for_status()

            data = response.json()

            return {
                "success": True,
                "generated_text": data.get(
                    "response",
                    "",
                ).strip(),
                "raw_response": data,
                "provider": "ollama",
                "model": self.ollama_model,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "provider": "ollama",
                "model": self.ollama_model,
            }

    def _generate_with_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:

        if not self.gemini_api_url:
            return {
                "success": False,
                "error": "Gemini API URL not configured.",
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

        if not self.gemini_api_key:
            return {
                "success": False,
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

            if response.status_code >= 400:
                return {
                    "success": False,
                    "error": (
                        f"Gemini API error {response.status_code}: "
                        f"{response.text[:500]}"
                    ),
                    "generated_text": "",
                    "provider": "gemini",
                    "model": self.gemini_model_name,
                }

            data = response.json()

            generated_text = self._extract_gemini_text(
                data,
            )

            if not generated_text:
                return {
                    "success": False,
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

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "generated_text": "",
                "provider": "gemini",
                "model": self.gemini_model_name,
            }

    def _build_gemini_url(
        self,
    ) -> str:

        separator = "&" if "?" in self.gemini_api_url else "?"

        return (
            f"{self.gemini_api_url}"
            f"{separator}key={self.gemini_api_key}"
        )

    def _extract_gemini_text(
        self,
        response_json: Dict[str, Any],
    ) -> str:

        try:
            return (
                response_json["candidates"][0]
                ["content"]["parts"][0]["text"]
                .strip()
            )

        except Exception:
            return ""