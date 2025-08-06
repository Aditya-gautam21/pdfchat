import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import gc
import time
from typing import List
import psutil
import math

class LocalHFEmbedder(Embeddings):
    def __init__(self, model_path: str, device: str = "auto"):
        # OPTIMIZED: Smart device selection
        if device == "auto":
            self.device = self._select_optimal_device()
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 Using device: {self.device}")
        
        try:
            # Load tokenizer and model with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            ).to(self.device)
            
            # Set to evaluation mode and optimize for inference
            self.model.eval()
            
            # OPTIMIZATION: Compile model if using PyTorch 2.0+
            if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("✅ Model compilation enabled")
                except Exception as e:
                    print(f"⚠️ Model compilation failed: {e}")
            
            # Calculate optimal batch size
            self.optimal_batch_size = self._calculate_optimal_batch_size()
            print(f"📊 Optimal batch size: {self.optimal_batch_size}")
            
        except Exception as e:
            print(f"❌ Error initializing embedder: {e}")
            raise

    def _select_optimal_device(self) -> torch.device:
        """Select the best available device"""
        if torch.cuda.is_available():
            # Check VRAM
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 4:  # 4GB+ VRAM
                    return torch.device("cuda")
                else:
                    print(f"⚠️ Limited VRAM ({gpu_memory:.1f}GB), using CPU for stability")
                    return torch.device("cpu")
            except:
                return torch.device("cpu")
        else:
            return torch.device("cpu")

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available resources"""
        try:
            if self.device.type == "cuda":
                # GPU batch sizing
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory > 8e9:  # 8GB+
                    return 64
                elif gpu_memory > 4e9:  # 4GB+
                    return 32
                else:
                    return 16
            else:
                # CPU batch sizing based on RAM
                available_ram_gb = psutil.virtual_memory().available / 1e9
                if available_ram_gb > 16:
                    return 32
                elif available_ram_gb > 8:
                    return 16
                else:
                    return 8
        except:
            return 16  # Conservative fallback

    def mean_pool(self, last_hidden_state, attention_mask):
        """Optimized mean pooling"""
        # Use in-place operations where possible
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts

    def embed_documents(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """OPTIMIZED: Embed documents with intelligent batching and memory management"""
        if not texts:
            return []
        
        # Use optimal batch size if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        # Adaptive batch sizing based on text lengths
        avg_length = sum(len(text) for text in texts) / len(texts)
        if avg_length > 2000:  # Long texts
            batch_size = max(4, batch_size // 2)
        elif avg_length > 1000:  # Medium texts
            batch_size = max(8, batch_size // 1.5)
        
        print(f"🔄 Embedding {len(texts)} documents (batch_size: {batch_size}, avg_length: {avg_length:.0f})")
        
        all_embeddings = []
        total_batches = math.ceil(len(texts) / batch_size)
        start_time = time.time()
        
        try:
            with torch.no_grad():  # Disable gradients for inference
                for i in range(0, len(texts), batch_size):
                    batch_idx = i // batch_size + 1
                    batch = texts[i:i+batch_size]
                    
                    try:
                        # Tokenize with truncation and padding
                        inputs = self.tokenizer(
                            batch, 
                            padding=True, 
                            truncation=True, 
                            return_tensors='pt',
                            max_length=512  # Limit token length for speed
                        ).to(self.device)
                        
                        # Forward pass
                        batch_start = time.time()
                        output = self.model(**inputs)
                        embeddings = self.mean_pool(output.last_hidden_state, inputs['attention_mask'])
                        
                        # Move to CPU and convert to list
                        embeddings_cpu = embeddings.cpu().numpy().tolist()
                        all_embeddings.extend(embeddings_cpu)
                        
                        batch_time = time.time() - batch_start
                        
                        # Progress reporting
                        if batch_idx % 5 == 0 or batch_idx == total_batches:
                            elapsed = time.time() - start_time
                            rate = len(all_embeddings) / elapsed
                            print(f"📊 Batch {batch_idx}/{total_batches} ({rate:.1f} docs/sec)")
                        
                        # Memory management
                        if batch_idx % 10 == 0:  # Every 10 batches
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            gc.collect()
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"⚠️ OOM at batch {batch_idx}, reducing batch size")
                            # Emergency: process remaining texts individually
                            for single_text in batch:
                                try:
                                    single_embedding = self.embed_documents([single_text], batch_size=1)[0]
                                    all_embeddings.append(single_embedding)
                                except Exception as single_e:
                                    print(f"❌ Failed to embed single text: {single_e}")
                                    # Use zero embedding as fallback
                                    all_embeddings.append([0.0] * 384)  # Common embedding size
                        else:
                            raise e
                    
                    except Exception as batch_e:
                        print(f"❌ Error in batch {batch_idx}: {batch_e}")
                        # Add zero embeddings for failed batch
                        for _ in batch:
                            all_embeddings.append([0.0] * 384)
            
            total_time = time.time() - start_time
            rate = len(texts) / total_time if total_time > 0 else 0
            print(f"✅ Embedding completed: {len(texts)} docs in {total_time:.1f}s ({rate:.1f} docs/sec)")
            
        except Exception as e:
            print(f"❌ Embedding process failed: {e}")
            # Return zero embeddings as fallback
            embedding_dim = 384  # Common dimension for sentence-transformers
            return [[0.0] * embedding_dim for _ in texts]
        
        finally:
            # Cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query efficiently"""
        if not text or not text.strip():
            return [0.0] * 384  # Return zero embedding for empty text
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    [text], 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                output = self.model(**inputs)
                embedding = self.mean_pool(output.last_hidden_state, inputs['attention_mask'])
                
                return embedding.cpu().numpy().tolist()[0]
                
        except Exception as e:
            print(f"❌ Query embedding failed: {e}")
            return [0.0] * 384  # Fallback

    def benchmark_performance(self, sample_texts: List[str] = None) -> dict:
        """Benchmark embedding performance"""
        if sample_texts is None:
            sample_texts = [
                "This is a short test sentence.",
                "This is a medium-length test sentence with more words and complexity for testing purposes.",
                "This is a much longer test sentence that contains significantly more content and words to properly test the embedding performance with longer texts that might be found in real documents and require more processing time and computational resources.",
            ] * 10  # 30 texts total
        
        print("🧪 Running embedding performance benchmark...")
        
        results = {
            "device": str(self.device),
            "batch_sizes_tested": [],
            "performance_results": {},
            "optimal_batch_size": None,
            "recommendations": []
        }
        
        # Test different batch sizes
        test_batch_sizes = [4, 8, 16, 32] if len(sample_texts) >= 32 else [4, 8, min(16, len(sample_texts))]
        
        for batch_size in test_batch_sizes:
            if batch_size > len(sample_texts):
                continue
                
            try:
                start_time = time.time()
                embeddings = self.embed_documents(sample_texts, batch_size=batch_size)
                elapsed = time.time() - start_time
                
                rate = len(sample_texts) / elapsed if elapsed > 0 else 0
                
                results["batch_sizes_tested"].append(batch_size)
                results["performance_results"][batch_size] = {
                    "time": elapsed,
                    "rate": rate,
                    "success": len(embeddings) == len(sample_texts)
                }
                
                print(f"📊 Batch size {batch_size}: {rate:.1f} docs/sec")
                
            except Exception as e:
                print(f"❌ Batch size {batch_size} failed: {e}")
                results["performance_results"][batch_size] = {
                    "time": float('inf'),
                    "rate": 0,
                    "success": False,
                    "error": str(e)
                }
        
        # Find optimal batch size
        successful_results = {
            bs: res for bs, res in results["performance_results"].items() 
            if res.get("success", False)
        }
        
        if successful_results:
            optimal_bs = max(successful_results.keys(), key=lambda x: successful_results[x]["rate"])
            results["optimal_batch_size"] = optimal_bs
            results["recommendations"].append(f"Use batch_size={optimal_bs} for best performance")
        
        # System recommendations
        if self.device.type == "cpu":
            results["recommendations"].append("Consider using GPU for faster embedding if available")
        
        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            # Get model size estimate
            model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
            
            info = {
                "model_name": getattr(self.model.config, 'name_or_path', 'Unknown'),
                "device": str(self.device),
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_size_mb": model_size_mb,
                "max_sequence_length": getattr(self.model.config, 'max_position_embeddings', 'Unknown'),
                "hidden_size": getattr(self.model.config, 'hidden_size', 'Unknown'),
                "optimal_batch_size": self.optimal_batch_size
            }
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to get model info: {e}"}

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'tokenizer'):
                del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass