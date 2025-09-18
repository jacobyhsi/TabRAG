import faiss
import json
import torch
import numpy as np

class VectorStore:
    def __init__(self, dim, index_type="Flat"):
        self.index = faiss.index_factory(dim, index_type)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadata):
        self.index.add(embeddings)
        self.texts.extend(texts if isinstance(texts, list) else [texts])
        self.metadata.extend(metadata if isinstance(metadata, list) else [metadata])

    def search(self, query_emb, k=5):
        D, I = self.index.search(query_emb, k)
        results = [
            {"text": self.texts[i], "meta": self.metadata[i], "score": float(D[0][j])}
            for j, i in enumerate(I[0])
        ]
        return results
    
    def save(self, path_prefix):
        faiss.write_index(self.index, f"{path_prefix}.index")
        with open(f"{path_prefix}_data.jsonl", "w") as f:
            for text, meta in zip(self.texts, self.metadata):
                f.write(json.dumps({"text": text, "meta": meta}) + "\n")

    @classmethod
    def load(cls, path_prefix):
        index = faiss.read_index(f"{path_prefix}.index")
        texts, metadata = [], []
        with open(f"{path_prefix}_data.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
                metadata.append(obj["meta"])
        store = cls(dim=index.d)
        store.index = index
        store.texts = texts
        store.metadata = metadata
        return store