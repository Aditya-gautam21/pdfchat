from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any
import requests

class OllamaLLM(LLM):
    model: str = "mistral"
    base_url: str = "http://localhost:11434"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()

        # Debug: check response structure
        if isinstance(result, dict):
            if "response" in result:
                return result["response"]
            elif "output" in result:
                return result["output"]
            elif "choices" in result and result["choices"]:
                return result["choices"][0].get("text", "")
            else:
                raise ValueError(f"Unexpected Ollama response keys: {list(result.keys())}")
        elif isinstance(result, str):
            return result
        else:
            raise TypeError(f"Unexpected result type: {type(result)}")

    @property
    def _llm_type(self) -> str:
        return "ollama"
