"""
Gemma Inference Backend supporting local Ollama execution on consumer hardware.
Provides sub-second local inference over HTTP REST API with zero external dependencies.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class OllamaGemmaBackend:
    """Interface to local Ollama server running Gemma models on Apple Silicon / GPU."""

    def __init__(
        self,
        model_name: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if Ollama server is running and model is loaded."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                return any(self.model_name in m for m in models)
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Sends generation request to local Ollama instance.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.95
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.base_url}: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation error: {e}")

    def __call__(self, prompt: str) -> str:
        """Callable wrapper for drop-in LLM backend compatibility."""
        return self.generate(prompt)
