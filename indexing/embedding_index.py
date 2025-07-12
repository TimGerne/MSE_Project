from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from typing import Any
import pandas as pd

# Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def build_embeddings(df: pd.DataFrame, model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    """
    Generates sentence embeddings for the 'main_content' column of the given DataFrame
    using the specified SentenceTransformer model.

    Args:
        df (pd.DataFrame): DataFrame containing a 'main_content' column with text content.
        model_name (str): Name of the sentence-transformer model to use.

    Returns:
        np.ndarray: A 2D NumPy array of shape (num_documents, embedding_dim) with float32 embeddings.
    """
    model = SentenceTransformer(model_name)
    texts = df["main_content"].fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> Any:
    """
    Builds a FAISS index using L2 (Euclidean) distance from the given embeddings.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of float32 shape (n_samples, dim).

    Returns:
        faiss.IndexFlatL2: A FAISS index populated with the given embeddings.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance
    index.add(embeddings)
    return index

def save_index(index: Any, index_path: str) -> None:
    """
    Saves the FAISS index to the specified file path.

    Args:
        index (faiss.IndexFlatL2): The FAISS index to save.
        index_path (str): Path to save the index file (typically with .faiss extension).
    """
    faiss.write_index(index, index_path)

def save_doc_mapping(df: pd.DataFrame, mapping_path: str) -> None:
    """
    Saves a document mapping (doc_id → {doc_id, url, title}) as a JSON file for later retrieval.

    Args:
        df (pd.DataFrame): DataFrame containing 'doc_id', 'url', and 'title' columns.
        mapping_path (str): Output path to store the JSON document mapping.
    """
    doc_mapping = {
        int(i): {
            "doc_id": int(row["doc_id"]),
            "url": row["url"],
            "title": row["title"]
        }
        for i, row in df.iterrows()
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(doc_mapping, f, ensure_ascii=False, indent=2)


