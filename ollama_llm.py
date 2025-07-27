import json
import requests
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM

class OllamaLLM(LLM):
    model: str = "mistral"
    base_url: str = "http://localhost:11434"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True
            },
            stream=True
        )

        output = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    token = data.get("response", "")
                    output += token
                except Exception as e:
                    print("Streaming error:", e)
                    continue
        return output

    @property
    def _llm_type(self) -> str:
        return "ollama"
