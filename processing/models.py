import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import faiss
import numpy as np
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from indexing.tokenize_utils import normalize_and_tokenize
from indexing.embedding_index import build_embeddings, build_faiss_index
from query_expansion import expand_query_with_prf, expand_query_with_synonyms, expand_query_with_filtered_synonyms

INDEX_DIR: str = "indexing/output"
FAISS_INDEX_PATH: str = f"{INDEX_DIR}/semantic_index.faiss"
DOC_MAPPING_PATH: str = f"{INDEX_DIR}/doc_mapping.json"


class BaseRetrievalModel(ABC):
    """
    Abstract base class for all retrieval models.
    """
    def __init__(self, texts: List[str], urls: List[str], use_expansion: bool = False) -> None:
        self.texts = texts
        self.urls = urls
        self.use_expansion = use_expansion
        self.doc_tokens: List[List[str]] = [normalize_and_tokenize(t) for t in texts]
        self.N: int = len(texts)

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top documents for a given query.

        Args:
            query (str): The user query.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Ranked list of retrieved documents with scores.
        """
        pass

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a string using normalize_and_tokenize.

        Args:
            text (str): Input text.

        Returns:
            List[str]: List of tokens.
        """
        return normalize_and_tokenize(text)


class BM25RetrievalModel(BaseRetrievalModel):
    """
    Custom BM25 retrieval model implementation.
    """
    def __init__(self, texts: List[str], urls: List[str], k1: float = 1.5, b: float = 0.75, use_expansion: bool = False) -> None:
        super().__init__(texts, urls, use_expansion=use_expansion)
        self.k1 = k1
        self.b = b
        self.use_expansion = use_expansion
        self.doc_lengths: List[int] = [len(doc) for doc in self.doc_tokens]
        self.avg_dl: float = np.mean(self.doc_lengths)
        self.vocabulary: set = set(term for doc in self.doc_tokens for term in doc)
        self.df: Counter = self._compute_df()
        self.idf: Dict[str, float] = self._compute_idf()

    def _compute_df(self) -> Counter:
        """
        Compute document frequency for each term.

        Returns:
            Counter: Document frequencies.
        """
        df = Counter()
        for doc in self.doc_tokens:
            for term in set(doc):
                df[term] += 1
        return df

    def _compute_idf(self) -> Dict[str, float]:
        """
        Compute inverse document frequency for each term.

        Returns:
            Dict[str, float]: IDF scores.
        """
        return {
            term: np.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def _score(self, query_tokens: List[str], doc_tokens: List[str], doc_len: int) -> float:
        """
        Compute BM25 score between query and document.

        Args:
            query_tokens (List[str]): Tokenized query.
            doc_tokens (List[str]): Tokenized document.
            doc_len (int): Length of document.

        Returns:
            float: BM25 score.
        """
        freqs = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * tf * (self.k1 + 1) / denom
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents for a given query using BM25.

        Args:
            query (str): Query string.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Ranked results with URL, score, snippet.
        """
        if self.use_expansion:
            #query = query = expand_query_with_filtered_synonyms(query, self.vocabulary)
            query = expand_query_with_prf(query, self.texts, top_k=10, num_terms=3)

        query_tokens = self.tokenize(query)
        scores = [
            (i, self._score(query_tokens, doc, len(doc)))
            for i, doc in enumerate(self.doc_tokens)
        ]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {"url": self.urls[i], "score": s, "snippet": self.texts[i][:300]}
            for i, s in ranked
        ] 


class TFIDFRetrievalModel(BaseRetrievalModel):
    """
    Custom TF-IDF retrieval model using scikit-learn vectorizer.
    """
    def __init__(self, texts: List[str], urls: List[str], use_expansion: bool = False) -> None:
        super().__init__(texts, urls, use_expansion=use_expansion)
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents using TF-IDF cosine similarity.

        Args:
            query (str): Query string.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Ranked results with URL, score, snippet.
        """
        if self.use_expansion:
            query = expand_query_with_synonyms(query)

        query_vec = self.vectorizer.transform([query])
        scores = (self.doc_matrix @ query_vec.T).toarray().flatten()
        ranked = np.argsort(-scores)[:top_k]
        return [
            {"url": self.urls[i], "score": scores[i], "snippet": self.texts[i][:300]}
            for i in ranked
        ]


class DenseRetrievalModel:
    """
    Dense vector retrieval using SentenceTransformer and FAISS.
    """
    def __init__(self,
                 faiss_index: faiss.IndexFlatIP,
                 doc_mapping: Dict[str, Dict[str, Any]],
                 model_name: str = "all-MiniLM-L6-v2", 
                 use_expansion: bool = False) -> None:
        self.index = faiss_index
        self.doc_mapping = doc_mapping
        self.model = SentenceTransformer(model_name)
        self.use_expansion = use_expansion

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k documents using dense embeddings and FAISS.

        Args:
            query (str): Query string.
            top_k (int): Number of results to return.

        Returns:
            List[Dict[str, Any]]: Ranked results with title, URL, score.
        """
        if self.use_expansion:
            query = expand_query_with_synonyms(query)

        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(query_vec, top_k)
        return [
            {
                "title": self.doc_mapping[str(i)]["title"],
                "url": self.doc_mapping[str(i)]["url"],
                "score": float(D[0][j])
            }
            for j, i in enumerate(I[0])
        ]


def load_faiss_and_mapping(index_path: str, mapping_path: str) -> Tuple[faiss.IndexFlatIP, Dict[str, Dict[str, Any]]]:
    """
    Load FAISS index and document mapping from disk.

    Args:
        index_path (str): Path to the FAISS index file.
        mapping_path (str): Path to the document mapping JSON.

    Returns:
        Tuple[faiss.IndexFlatIP, Dict[str, Dict[str, Any]]]: Loaded index and mapping.
    """
    faiss_index = faiss.read_index(index_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        doc_mapping = json.load(f)
    return faiss_index, doc_mapping
