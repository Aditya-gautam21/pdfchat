import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel

class LocalHFEmbedder(Embeddings):
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        return summed / counts
    
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                output = self.model(**inputs)
                embeddings = self.mean_pool(output.last_hidden_state, inputs['attention_mask'])
            all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return all_embeddings


    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]