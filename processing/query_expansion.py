from nltk.corpus import wordnet
from indexing.tokenize_utils import normalize_and_tokenize
import numpy as np
from collections import Counter
from typing import List


def expand_query_with_synonyms(query: str, max_synonyms=2) -> str:
    expanded_terms = []
    for word in query.split():
        synonyms = wordnet.synsets(word)
        lemmas = set()
        for syn in synonyms:
            for lemma in syn.lemmas():
                lemmas.add(lemma.name().replace("_", " "))
                if len(lemmas) >= max_synonyms:
                    break
            if len(lemmas) >= max_synonyms:
                break
        expanded_terms.extend(list(lemmas))
    expanded_query = query + ' ' + ' '.join(expanded_terms)
    return expanded_query

def expand_query_with_filtered_synonyms(query: str, vocabulary: set, max_synonyms: int = 2) -> str:
    expanded_terms = []
    tokens = normalize_and_tokenize(query)

    for token in tokens:
        synonyms = {
            lemma.name().replace('_', ' ')
            for syn in wordnet.synsets(token)
            for lemma in syn.lemmas()
        }
        filtered = [
            syn for syn in synonyms
            if syn.lower() != token and syn.lower() in vocabulary
        ][:max_synonyms]  # Limit synonyms per word
        expanded_terms.extend(filtered)

    return query + " " + " ".join(expanded_terms)

def expand_query_with_prf(query: str, texts: List[str], top_k: int = 10, num_terms: int = 3) -> str:
    """
    Expand query using Pseudo-Relevance Feedback (PRF).

    Args:
        query (str): Original user query.
        texts (List[str]): Corpus documents (same as passed to BM25).
        top_k (int): Number of top documents to use for feedback.
        num_terms (int): Number of expansion terms to add.

    Returns:
        str: Expanded query.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Fit TF-IDF on the entire corpus
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)

    # Transform query
    query_vec = vectorizer.transform([query])
    scores = (doc_matrix @ query_vec.T).toarray().flatten()

    # Select top_k documents
    top_doc_indices = np.argsort(-scores)[:top_k]
    token_counter = Counter()

    for idx in top_doc_indices:
        tokens = normalize_and_tokenize(texts[idx])
        token_counter.update(tokens)

    # Remove query terms from expansion
    query_tokens = set(normalize_and_tokenize(query))
    candidate_terms = [term for term, _ in token_counter.most_common()
                       if term not in query_tokens and len(term) > 2]

    expansion_terms = candidate_terms[:num_terms]
    expanded_query = query + " " + " ".join(expansion_terms)

    return expanded_query