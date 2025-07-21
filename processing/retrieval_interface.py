import pandas as pd
from typing import List
from models import BM25RetrievalModel, DenseRetrievalModel, HybridAlphaModel, HybridReciprocalRankFusionModel,load_faiss_and_mapping


def generate_results_dataframe(queries: List[str], model_name: str = "bm25", use_expansion: bool = False, alpha: float = 0.5, top_k: int = 100) -> pd.DataFrame:
    """
    Takes a list of queries and returns a Pandas DataFrame containing the 
    top-k retrieval results (with rank, URL, and score) for each query using a 
    selected retrieval model
    """
    faiss_index, doc_mapping = load_faiss_and_mapping(
        "Users/lilieven/Documents/Uni/Master/Semester_4/ModernSearchEngines/MSE_Project/indexing/indexing/output/semantic_index.faiss",
        "Users/lilieven/Documents/Uni/Master/Semester_4/ModernSearchEngines/MSE_Project/indexing/indexing/output/doc_mapping.json"
    )
    texts = [doc["title"] for doc in doc_mapping.values()]
    urls = [doc["url"] for doc in doc_mapping.values()]

    # Init models
    if model_name == "bm25":
        model = BM25RetrievalModel(texts, urls, use_expansion=use_expansion)
    elif model_name == "dense":
        model = DenseRetrievalModel(faiss_index, doc_mapping, use_expansion=use_expansion)
    elif model_name == "hybrid_rrf":
        bm25 = BM25RetrievalModel(texts, urls, use_expansion=use_expansion)
        dense = DenseRetrievalModel(faiss_index, doc_mapping, use_expansion=use_expansion)
        model = HybridReciprocalRankFusionModel(bm25, dense, doc_mapping=doc_mapping)
    elif model_name == "hybrid_alpha":
        bm25 = BM25RetrievalModel(texts, urls, use_expansion=use_expansion)
        dense = DenseRetrievalModel(faiss_index, doc_mapping, use_expansion=use_expansion)
        model = HybridAlphaModel(bm25, dense, alpha=alpha)
    else:
        raise ValueError("Unsupported model name.")

    # Prepare results
    records = []
    for i, query in enumerate(queries):
        results = model.retrieve(query, top_k=top_k)
        for rank, item in enumerate(results, 1):
            records.append({
                "Query": i,
                "Rank": rank,
                "URL": item["url"],
                "Relevance": item.get("score", 0.0)
            })
    return pd.DataFrame.from_records(records)
