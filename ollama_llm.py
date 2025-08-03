import json
import requests
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM

class OllamaLLM(LLM):
    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        try:
            # Check if server is running first
            health_response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if health_response.status_code != 200:
                return "❌ Error: Ollama server is not responding properly. Please restart with 'ollama serve'"
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature
                    }
                },
                stream=True,
                timeout=60
            )
            response.raise_for_status()
            
        except requests.exceptions.ConnectionError:
            return "❌ Error: Cannot connect to Ollama server. Please start it with 'ollama serve'"
        except requests.exceptions.Timeout:
            return "❌ Error: Request timed out. Ollama server might be overloaded."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return f"❌ Error: Model '{self.model}' not found. Please install it with 'ollama pull {self.model}'"
            return f"❌ Error: HTTP {e.response.status_code} - {str(e)}"
        except Exception as e:
            return f"❌ Error: {str(e)}"
        
        output = ""
        try:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            output += data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            return f"❌ Error processing response: {str(e)}"
        
        return output if output else "❌ No response received"

    @property
    def _llm_type(self) -> str:
        return "ollama"
