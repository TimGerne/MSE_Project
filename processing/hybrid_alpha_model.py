from typing import List, Dict, Any

class HybridAlphaModel:
    def __init__(self, bm25_model, dense_model, alpha: float = 0.5):
        self.bm25_model = bm25_model
        self.dense_model = dense_model
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_model.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_model.retrieve(query, top_k=top_k * 2)

        # Map URL to score
        bm25_scores = {r['url']: r.get('score', 1.0) for r in bm25_results}
        dense_scores = {r['url']: r.get('score', 1.0) for r in dense_results}

        # Combine all URLs
        all_urls = set(bm25_scores) | set(dense_scores)
        combined = []
        for url in all_urls:
            bm25 = bm25_scores.get(url, 0.0)
            dense = dense_scores.get(url, 0.0)
            score = self.alpha * bm25 + (1 - self.alpha) * dense
            combined.append({
                "url": url,
                "score": score
            })

        combined.sort(key=lambda x: x['score'], reverse=True)
        return combined[:top_k]
