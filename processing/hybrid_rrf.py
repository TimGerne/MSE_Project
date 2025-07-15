from collections import defaultdict
from typing import List, Dict, Any
from models import BM25RetrievalModel, DenseRetrievalModel


class HybridReciprocalRankFusionModel:
    """
    Hybrid retrieval model using Reciprocal Rank Fusion (RRF) to combine BM25 and Dense retrieval.
    """
    def __init__(self,
                 bm25_model: BM25RetrievalModel,
                 dense_model: DenseRetrievalModel,
                 k: int = 60) -> None:
        self.bm25_model = bm25_model
        self.dense_model = dense_model
        self.k = k  # RRF constant

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents by combining BM25 and Dense rankings using RRF.

        Args:
            query (str): Query string.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Ranked results with title, URL, combined RRF score.
        """
        bm25_results = self.bm25_model.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_model.retrieve(query, top_k=top_k * 2)

        # Calculate Reciprocal Rank scores
        rrf_scores = defaultdict(float)

        def update_rrf(results: List[Dict[str, Any]]) -> None:
            for rank, item in enumerate(results):
                url = item["url"]
                rrf_scores[url] += 1.0 / (self.k + rank + 1)

        update_rrf(bm25_results)
        update_rrf(dense_results)

        # Combine metadata
        doc_lookup = {item["url"]: item for item in bm25_results + dense_results}

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "url": url,
                "score": score,
                "title": doc_lookup[url].get("title", ""),
                "snippet": doc_lookup[url].get("snippet", "") or doc_lookup[url].get("title", "")
            }
            for url, score in ranked
        ]
