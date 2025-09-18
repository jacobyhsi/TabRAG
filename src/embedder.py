from abc import ABC, abstractmethod
import numpy as np

from sentence_transformers import SentenceTransformer
import openai

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