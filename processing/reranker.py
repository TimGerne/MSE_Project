from sentence_transformers import CrossEncoder
from typing import List, Dict

class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Dict[str, str]], top_k: int = 10) -> List[Dict[str, str]]:
        pairs = [(query, doc["snippet"]) for doc in docs]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            docs[i]["score"] = float(score)

        reranked = sorted(docs, key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]
