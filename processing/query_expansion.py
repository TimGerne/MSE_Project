from nltk.corpus import wordnet
from indexing.tokenize_utils import normalize_and_tokenize
import numpy as np
from collections import Counter
from typing import List
from gensim.downloader import load as gensim_load
from sklearn.feature_extraction.text import TfidfVectorizer

glove_model = gensim_load("glove-wiki-gigaword-100") 

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
    Expand query using Pseudo-Relevance Feedback
    """
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)

    query_vec = vectorizer.transform([query])
    scores = (doc_matrix @ query_vec.T).toarray().flatten()

    top_doc_indices = np.argsort(-scores)[:top_k]
    token_counter = Counter()

    for idx in top_doc_indices:
        tokens = normalize_and_tokenize(texts[idx])
        token_counter.update(tokens)

    query_tokens = set(normalize_and_tokenize(query))
    candidate_terms = [term for term, _ in token_counter.most_common()
                       if term not in query_tokens and len(term) > 2]

    expansion_terms = candidate_terms[:num_terms]
    expanded_query = query + " " + " ".join(expansion_terms)

    return expanded_query


def expand_query_with_glove(query: str, vocabulary: set, max_expansions_per_term: int = 2) -> str:
    query_tokens = normalize_and_tokenize(query)
    expanded = set(query_tokens)

    for token in query_tokens:
        if token not in glove_model:
            continue
        similar_words = glove_model.most_similar(token, topn=10)
        for similar_word, _ in similar_words:
            if similar_word in vocabulary and similar_word not in expanded:
                expanded.add(similar_word)
            if len(expanded) - len(query_tokens) >= max_expansions_per_term:
                break

    return ' '.join(expanded)