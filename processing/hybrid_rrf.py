from collections import defaultdict
from typing import List, Dict, Any
from models import BM25RetrievalModel, DenseRetrievalModel

class HybridReciprocalRankFusionModel:
    def __init__(self,
                 bm25_model: BM25RetrievalModel,
                 dense_model: DenseRetrievalModel,
                 doc_mapping: Dict[str, Dict[str, Any]],
                 k: int = 40) -> None:
        self.bm25_model = bm25_model
        self.dense_model = dense_model
        self.doc_mapping = doc_mapping
        self.k = k
        self.url_to_doc_id = {
            doc["url"]: doc_id for doc_id, doc in doc_mapping.items()
        }

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        #print(f"[HybridRRF] Using RRF k = {self.k}")
        bm25_results = self.bm25_model.retrieve(query, top_k=top_k * 10)
        dense_results = self.dense_model.retrieve(query, top_k=top_k * 10)

        rrf_scores = defaultdict(float)

        def update_rrf(results: List[Dict[str, Any]]):
            for rank, item in enumerate(results):
                url = item["url"]
                rrf_scores[url] += 1.0 / (self.k + rank + 1)

        update_rrf(bm25_results)
        update_rrf(dense_results)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "url": url,
                "score": score,
                "title": self.doc_mapping.get(self.url_to_doc_id.get(url, ""), {}).get("title", ""),
                "snippet": self.doc_mapping.get(self.url_to_doc_id.get(url, ""), {}).get("main_content", "")[:500]
            }
            for url, score in ranked
        ]
