import asyncio
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, index_file, data_file, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index_file = Path(index_file)
        self.data_file = Path(data_file)
        self.index = faiss.read_index(str(self.index_file))
        with open(self.data_file) as f:
            self.data = json.load(f)

    async def search(self, query, top_k=5):
        query_vec = await asyncio.to_thread(
            self.model.encode, [query], convert_to_numpy=True
        )
        distances, indices = self.index.search(query_vec, top_k)
        results = [
            {"chunk": self.data[i], "score": float(distances[0][j])}
            for j, i in enumerate(indices[0])
            if i >= 0
        ]
        return results
