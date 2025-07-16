from collections import defaultdict
import numpy as np
from models import HybridRetrievalModel
import matplotlib.pyplot as plt

def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            qid, _, url, rel = line.strip().split()
            if int(rel) > 0:
                qrels[qid].add(url)
    return qrels

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return sum(1 for url in retrieved_k if url in relevant) / k

def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not relevant:
        return 0.0
    return sum(1 for url in retrieved_k if url in relevant) / len(relevant)

def ndcg_at_k(retrieved, relevant, k):
    dcg = 0.0
    for i, url in enumerate(retrieved[:k]):
        if url in relevant:
            dcg += 1 / np.log2(i + 2)
    ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def evaluate(model, queries_path, qrels_path, k=10):
    qrels = load_qrels(qrels_path)
    with open(queries_path, 'r') as f:
        lines = f.readlines()

    metrics = {'precision': [], 'recall': [], 'ndcg': []}

    for line in lines:
        qid, query = line.strip().split('\t')
        relevant = qrels.get(qid, set())

        results = model.retrieve(query, top_k=k)
        retrieved_urls = [r['url'] for r in results]

        metrics['precision'].append(precision_at_k(retrieved_urls, relevant, k))
        metrics['recall'].append(recall_at_k(retrieved_urls, relevant, k))
        metrics['ndcg'].append(ndcg_at_k(retrieved_urls, relevant, k))

    return {
        f'precision@{k}': np.mean(metrics['precision']),
        f'recall@{k}': np.mean(metrics['recall']),
        f'ndcg@{k}': np.mean(metrics['ndcg']),
    }


def tune_hybrid_alpha(bm25_model, dense_model, queries_path, qrels_path, k=10):

    best_alpha = None
    best_score = -1
    results = []

    for alpha in np.linspace(0, 1, 11):  # Try alpha = 0.0 to 1.0
        hybrid = HybridRetrievalModel(bm25_model, dense_model, alpha=alpha)
        metrics = evaluate(hybrid, queries_path, qrels_path, k=k)
        ndcg = metrics[f'ndcg@{k}']
        results.append((alpha, ndcg))

        print(f"Alpha={alpha:.2f} → NDCG@{k} = {ndcg:.4f}")

        if ndcg > best_score:
            best_score = ndcg
            best_alpha = alpha

    print(f"\n Best Alpha: {best_alpha:.2f} with NDCG@{k} = {best_score:.4f}")
    return best_alpha, results

# Plotting
def plot_alpha_scores(results):
    alphas, ndcgs = zip(*results)
    plt.plot(alphas, ndcgs, marker='o')
    plt.title('NDCG@k vs Hybrid Alpha')
    plt.xlabel('Alpha (BM25 weight)')
    plt.ylabel('NDCG@k')
    plt.grid(True)
    plt.show()