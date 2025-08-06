import json
import requests
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM
import time
import threading
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class OllamaLLM(LLM):
    model: str = "mistral:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    session: None = None
    base_timeout: None = None
    max_timeout: None = None
    timeout_multiplier: None = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # OPTIMIZED: Configure session with retry strategy and connection pooling
        self.session = requests.Session()
        
        # Retry strategy for transient failures
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=2, pool_maxsize=5)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # TIMEOUT MANAGEMENT: Progressive timeout strategy
        self.base_timeout = 30  # Start with 30s
        self.max_timeout = 120  # Max 2 minutes
        self.timeout_multiplier = 1.5
        
        # CONNECTION POOL: Keep connection warm
        self._warm_connection()

    def _warm_connection(self):
        """Pre-warm the connection to Ollama server"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("🔗 Connection to Ollama server established")
            else:
                print(f"⚠️ Ollama server responded with status {response.status_code}")
        except Exception as e:
            print(f"⚠️ Could not establish initial connection: {e}")

    def _adaptive_timeout(self, prompt_length: int) -> int:
        """Calculate timeout based on prompt complexity"""
        # Base timeout + additional time for longer prompts
        base_time = 30
        
        if prompt_length > 5000:  # Very long prompt (table data)
            return min(120, base_time + 60)
        elif prompt_length > 2000:  # Medium prompt
            return min(90, base_time + 30)
        else:  # Short prompt
            return base_time

    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt to reduce processing time while maintaining accuracy"""
        # Remove excessive whitespace that doesn't affect meaning
        optimized = ' '.join(prompt.split())
        
        # For very long prompts, add processing hints
        if len(optimized) > 4000:
            hints = "\n\nIMPORTANT: Focus on the specific question asked. Be concise but complete in your answer."
            optimized = optimized + hints
        
        return optimized

    def _check_server_health(self) -> tuple[bool, str]:
        """Quick health check with detailed diagnostics"""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                server_info = response.json()
                models = server_info.get('models', [])
                model_names = [model.get('name', '') for model in models]
                
                if self.model not in model_names:
                    return False, f"Model '{self.model}' not found. Available: {', '.join(model_names)}"
                
                if response_time > 3:
                    return True, f"Server slow (response: {response_time:.1f}s) but functional"
                
                return True, f"Server healthy (response: {response_time:.1f}s)"
            else:
                return False, f"Server returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama server. Please start with 'ollama serve'"
        except requests.exceptions.Timeout:
            return False, "Health check timed out. Server may be overloaded"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # PHASE 1: Health check and optimization
        server_healthy, health_message = self._check_server_health()
        if not server_healthy:
            return f"❌ Error: {health_message}"
        
        # PHASE 2: Prompt optimization and timeout calculation
        optimized_prompt = self._optimize_prompt(prompt)
        timeout_duration = self._adaptive_timeout(len(optimized_prompt))
        
        print(f"🔄 Processing request (timeout: {timeout_duration}s, prompt: {len(optimized_prompt)} chars)")
        
        # PHASE 3: Request with retry logic and progressive timeout
        max_retries = 2
        current_timeout = timeout_duration
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt}/{max_retries}")
                    current_timeout = min(self.max_timeout, int(current_timeout * self.timeout_multiplier))
                
                # OPTIMIZED: Request configuration for table/data processing
                request_data = {
                    "model": self.model,
                    "prompt": optimized_prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.temperature,
                        "top_p": 0.1,  # Very focused for structured data
                        "top_k": 10,   # Limited choices for accuracy
                        "repeat_penalty": 1.1,
                        "num_predict": 1024,  # Reasonable response length
                        "num_ctx": 4096,      # Context window
                        "num_batch": 512,     # Batch size for processing
                        "num_keep": 24,       # Keep tokens for consistency
                    }
                }
                
                # STREAMING: Process response with timeout and progress tracking
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    stream=True,
                    timeout=current_timeout
                )
                response.raise_for_status()
                
                # PHASE 4: Stream processing with timeout protection
                return self._process_streaming_response(response, current_timeout)
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    print(f"⏱️ Request timed out after {current_timeout}s, retrying with longer timeout...")
                    time.sleep(2)  # Brief pause before retry
                    continue
                else:
                    return f"❌ Error: Request timed out after {current_timeout}s. Try a shorter query or check server load."
                    
            except requests.exceptions.ConnectionError:
                if attempt == 0:  # Only try to reconnect once
                    print("🔄 Connection lost, attempting to reconnect...")
                    self._warm_connection()
                    time.sleep(1)
                    continue
                else:
                    return "❌ Error: Lost connection to Ollama server. Please restart with 'ollama serve'"
                    
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    return f"❌ Error: Model '{self.model}' not found. Please install it with 'ollama pull {self.model}'"
                elif e.response.status_code == 429:
                    if attempt < max_retries:
                        wait_time = (attempt + 1) * 2
                        print(f"⏱️ Server busy, waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return "❌ Error: Server is overloaded. Please try again later."
                else:
                    return f"❌ Error: HTTP {e.response.status_code} - {str(e)}"
                    
            except Exception as e:
                if attempt < max_retries and "timeout" in str(e).lower():
                    print(f"⚠️ Unexpected timeout, retrying...")
                    continue
                else:
                    return f"❌ Error: {str(e)}"
        
        return "❌ Error: All retry attempts failed"

    def _process_streaming_response(self, response, timeout_duration: int) -> str:
        """Process streaming response with timeout and progress tracking"""
        output = ""
        start_time = time.time()
        last_progress_time = start_time
        
        try:
            for line in response.iter_lines():
                current_time = time.time()
                
                # Check for overall timeout
                if current_time - start_time > timeout_duration:
                    print(f"⏱️ Stream processing timed out after {timeout_duration}s")
                    if output:
                        return output + "\n\n[Response truncated due to timeout]"
                    else:
                        return "❌ Error: Stream processing timed out before receiving response"
                
                # Progress indicator for long responses
                if current_time - last_progress_time > 10:  # Every 10 seconds
                    elapsed = current_time - start_time
                    print(f"🔄 Processing... ({elapsed:.1f}s elapsed, {len(output)} chars received)")
                    last_progress_time = current_time
                
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            chunk = data["response"]
                            output += chunk
                            
                            # OPTIMIZATION: Early termination for very long responses
                            if len(output) > 8192:  # 8KB limit
                                print("⚠️ Response length limit reached, terminating stream")
                                output += "\n\n[Response truncated - limit reached]"
                                break
                        
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                    except UnicodeDecodeError:
                        continue
        
        except requests.exceptions.RequestException as e:
            if output:
                return output + f"\n\n[Stream interrupted: {str(e)}]"
            else:
                return f"❌ Error: Stream processing failed - {str(e)}"
        
        except Exception as e:
            if output:
                return output + f"\n\n[Processing error: {str(e)}]"
            else:
                return f"❌ Error: Unexpected error in stream processing - {str(e)}"
        
        # Final validation
        if not output or not output.strip():
            return "❌ Error: No response received from server"
        
        processing_time = time.time() - start_time
        print(f"✅ Response completed in {processing_time:.1f}s ({len(output)} characters)")
        
        return output.strip()

    def _estimate_processing_time(self, prompt_length: int) -> int:
        """Estimate processing time based on prompt complexity"""
        # Base processing time estimates (in seconds)
        base_time = 5
        
        # Time per character (very rough estimate)
        time_per_char = 0.01
        
        # Additional time for complex operations
        if "table" in prompt_length or "data" in str(prompt_length):
            complexity_multiplier = 1.5
        else:
            complexity_multiplier = 1.0
        
        estimated = base_time + (prompt_length * time_per_char * complexity_multiplier)
        return min(120, max(15, int(estimated)))  # Between 15s and 2min

    def test_connection(self) -> dict:
        """Comprehensive connection test for debugging"""
        results = {
            "server_reachable": False,
            "model_available": False,
            "response_time": None,
            "server_info": None,
            "test_query_success": False,
            "recommendations": []
        }
        
        try:
            # Test 1: Server reachability
            start_time = time.time()
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            response_time = time.time() - start_time
            
            results["response_time"] = response_time
            
            if response.status_code == 200:
                results["server_reachable"] = True
                server_data = response.json()
                results["server_info"] = server_data
                
                # Test 2: Model availability
                models = [model.get("name", "") for model in server_data.get("models", [])]
                results["model_available"] = self.model in models
                
                if not results["model_available"]:
                    results["recommendations"].append(f"Install model: ollama pull {self.model}")
                
                # Test 3: Simple query
                if results["model_available"]:
                    try:
                        test_response = self._call("Hello", stop=None)
                        results["test_query_success"] = not test_response.startswith("❌")
                    except:
                        results["test_query_success"] = False
                
            else:
                results["recommendations"].append("Check if Ollama server is running properly")
                
        except requests.exceptions.ConnectionError:
            results["recommendations"].append("Start Ollama server with 'ollama serve'")
        except requests.exceptions.Timeout:
            results["recommendations"].append("Server is responding slowly - check system resources")
        except Exception as e:
            results["recommendations"].append(f"Unexpected error: {str(e)}")
        
        # Performance recommendations
        if results["response_time"] and results["response_time"] > 5:
            results["recommendations"].append("Server is slow - consider system optimization")
        
        return results

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def __del__(self):
        """Cleanup session on destruction"""
        if hasattr(self, 'session'):
            try:
                self.session.close()
            except:
                pass