import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

class BaseRetrievalModel:
    def __init__(self, texts, urls):
        self.texts = texts
        self.urls = urls
        self.N = len(texts)
        self.doc_tokens = [self.tokenize(doc) for doc in texts]

    def tokenize(self, text):
        return word_tokenize(text.lower())

    def retrieve(self, query, top_k=100):
        raise NotImplementedError


class BM25RetrievalModel(BaseRetrievalModel):
    def __init__(self, texts, urls, k1=1.5, b=0.75):
        super().__init__(texts, urls)
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in self.doc_tokens]
        self.avg_dl = np.mean(self.doc_lengths)
        self.df = self._compute_df()
        self.idf = self._compute_idf()

    def _compute_df(self):
        df = Counter()
        for doc in self.doc_tokens:
            for term in set(doc):
                df[term] += 1
        return df

    def _compute_idf(self):
        idf = {}
        for term, freq in self.df.items():
            idf[term] = np.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
        return idf

    def _score(self, query_tokens, doc_tokens, doc_len):
        freqs = Counter(doc_tokens)
        score = 0.0
        for term in query_tokens:
            if term not in freqs:
                continue
            tf = freqs[term]
            idf = self.idf.get(term, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            score += idf * tf * (self.k1 + 1) / denom
        return score

    def retrieve(self, query, top_k=100):
        query_tokens = self.tokenize(query)
        scores = [(i, self._score(query_tokens, doc, len(doc)))
                  for i, doc in enumerate(self.doc_tokens)]
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        return [{'url': self.urls[i], 'score': s, 'snippet': self.texts[i][:300]}
                for i, s in ranked]


class TFIDFRetrievalModel(BaseRetrievalModel):
    def __init__(self, texts, urls):
        super().__init__(texts, urls)
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query, top_k=100):
        query_vec = self.vectorizer.transform([query])
        scores = self.doc_matrix @ query_vec.T
        ranked = np.argsort(-scores.toarray().flatten())[:top_k]
        return [{'url': self.urls[i], 'score': scores[i, 0], 'snippet': self.texts[i][:300]}
                for i in ranked]


class DenseRetrievalModel(BaseRetrievalModel):
    def __init__(self, texts, urls, model_name="paraphrase-MiniLM-L6-v2"):
        super().__init__(texts, urls)
        self.encoder = SentenceTransformer(model_name)
        self.doc_embeddings = self.encoder.encode(texts, convert_to_tensor=False)

    def retrieve(self, query, top_k=100):
        query_embedding = self.encoder.encode([query], convert_to_tensor=False)
        sims = cosine_similarity(query_embedding, self.doc_embeddings)[0]
        top_indices = np.argsort(-sims)[:top_k]
        return [
            {
                'url': self.urls[i],
                'score': sims[i],
                'snippet': self.texts[i][:300]
            }
            for i in top_indices
        ]
        

class HybridRetrievalModel(BaseRetrievalModel):
    def __init__(self, bm25_model, dense_model, alpha=0.5):
        self.bm25 = bm25_model
        self.dense = dense_model
        self.alpha = alpha  # weight between BM25 and dense

    def retrieve(self, query, top_k=100):
        bm25_results = self.bm25.retrieve(query, top_k=None)
        dense_results = self.dense.retrieve(query, top_k=None)

        # Convert to dictionaries: url -> score
        bm25_scores = {r['url']: r['score'] for r in bm25_results}
        dense_scores = {r['url']: r['score'] for r in dense_results}

        # Union of URLs
        all_urls = set(bm25_scores) | set(dense_scores)

        # Normalize both sets of scores
        def normalize(score_dict):
            urls = list(score_dict.keys())
            scores = np.array(list(score_dict.values())).reshape(-1, 1)
            scaler = MinMaxScaler()
            norm_scores = scaler.fit_transform(scores).flatten()
            return dict(zip(urls, norm_scores))

        norm_bm25 = normalize(bm25_scores)
        norm_dense = normalize(dense_scores)

        # Weighted combination
        final_scores = {
            url: self.alpha * norm_bm25.get(url, 0) + (1 - self.alpha) * norm_dense.get(url, 0)
            for url in all_urls
        }

        # Sort and return top_k
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            {
                'url': url,
                'score': score,
                'snippet': self.bm25.texts[self.bm25.urls.index(url)][:300]
            }
            for url, score in ranked
        ]
        
class HybridRetrievalModel(BaseRetrievalModel):
    def __init__(self, bm25_model, dense_model, alpha=0.5):
        self.bm25 = bm25_model
        self.dense = dense_model
        self.alpha = alpha  # weight between BM25 and dense

    def retrieve(self, query, top_k=100):
        bm25_results = self.bm25.retrieve(query, top_k=None)
        dense_results = self.dense.retrieve(query, top_k=None)

        # Convert to dictionaries: url -> score
        bm25_scores = {r['url']: r['score'] for r in bm25_results}
        dense_scores = {r['url']: r['score'] for r in dense_results}

        # Union of URLs
        all_urls = set(bm25_scores) | set(dense_scores)

        # Normalize both sets of scores
        def normalize(score_dict):
            urls = list(score_dict.keys())
            scores = np.array(list(score_dict.values())).reshape(-1, 1)
            scaler = MinMaxScaler()
            norm_scores = scaler.fit_transform(scores).flatten()
            return dict(zip(urls, norm_scores))

        norm_bm25 = normalize(bm25_scores)
        norm_dense = normalize(dense_scores)

        # Weighted combination
        final_scores = {
            url: self.alpha * norm_bm25.get(url, 0) + (1 - self.alpha) * norm_dense.get(url, 0)
            for url in all_urls
        }

        # Sort and return top_k
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])[:top_k]
        return [
            {
                'url': url,
                'score': score,
                'snippet': self.bm25.texts[self.bm25.urls.index(url)][:300]
            }
            for url, score in ranked
        ]
