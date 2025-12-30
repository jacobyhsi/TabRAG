from abc import ABC, abstractmethod
import openai
import vllm
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

class BaseEmbedder(ABC):
    @abstractmethod
    def get_dims(self) -> int:
        pass

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        pass

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.emb_dim = self.model.get_sentence_embedding_dimension()

    def get_dims(self) -> int:
        return self.emb_dim
    
    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.emb_dim = self.model.config.hidden_size

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def get_dims(self) -> int:
        return self.emb_dim
    
    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

class VLLMEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-8B", **kwargs):
        self.model = vllm.LLM(model=model_name, **kwargs)

        dummy_output = self.model.embed("hello")
        self.emb_dim = len(dummy_output[0].outputs.embedding)

    def get_dims(self) -> int:
        return self.emb_dim

    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        outputs = self.model.embed(texts)
        embeddings_list = [o.outputs.embedding for o in outputs]
        embeddings_tensor = torch.tensor(embeddings_list)
        normalized_embeddings = F.normalize(embeddings_tensor, p=2, dim=1)
        
        return normalized_embeddings.cpu().numpy()

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name):
        self.model_name = model_name
        response = openai.Embedding.create(
            model=self.model_name,
            input=["hello world"]
        )
        embedding = response['data'][0]['embedding']
        self.emb_dim = len(embedding)

    def get_dims(self) -> int:
        return self.emb_dim
    
    def encode(self, texts: list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        response = openai.Embedding.create(
            model=self.model_name, 
            input=texts)
        embeddings = [r["embedding"] for r in response["data"]]
        return np.array(embeddings)