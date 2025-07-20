from collections import defaultdict
import numpy as np
from hybrid_rrf import HybridReciprocalRankFusionModel
import matplotlib.pyplot as plt
import argparse
from models import BM25RetrievalModel, DenseRetrievalModel, load_faiss_and_mapping
import json
from hybrid_alpha_model import HybridAlphaModel
import time
from reranker import CrossEncoderReranker

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

def evaluate(model, queries_path, qrels_path, k=100):
    qrels = load_qrels(qrels_path)
    with open(queries_path, 'r') as f:
        lines = f.readlines()

    metrics = {'precision': [], 'recall': [], 'ndcg': []}
    start_time = time.time()

    # for line in lines:
    #     qid, query = line.strip().split('\t')
    #     relevant = qrels.get(qid, set())

    #     results = model.retrieve(query, top_k=k)
    #     retrieved_urls = [r['url'] for r in results]

    #     metrics['precision'].append(precision_at_k(retrieved_urls, relevant, k))
    #     metrics['recall'].append(recall_at_k(retrieved_urls, relevant, k))
    #     metrics['ndcg'].append(ndcg_at_k(retrieved_urls, relevant, k))
    with open("retrieved_urls_log.txt", "w", encoding="utf-8") as out:
        for line in lines:
            qid, query = line.strip().split('\t')
            relevant = qrels.get(qid, set())

            results = model.retrieve(query, top_k=100)  # retrieve more for reranking
            if args.use_reranker:
                try:
                    from reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker()
                    results = reranker.rerank(query, results, top_k=k)
                except Exception as e:
                    print(f"[⚠️ Reranker Error] {e} – using original results")

            retrieved_urls = [r['url'] for r in results]

            out.write(f"\nQuery {qid}: {query}\n")
            for i, r in enumerate(results, 1):
                mark = "✅" if r['url'] in relevant else ""
                out.write(f"{i}. {r['url']} {mark}\n")

            metrics['precision'].append(precision_at_k(retrieved_urls, relevant, k))
            metrics['recall'].append(recall_at_k(retrieved_urls, relevant, k))
            metrics['ndcg'].append(ndcg_at_k(retrieved_urls, relevant, k))
    
    elapsed_time = time.time() - start_time
    print(f"\n Evaluation took {elapsed_time:.2f} seconds for {len(lines)} queries")

    # Plot metrics
    avg_precision = np.mean(metrics['precision'])
    avg_recall = np.mean(metrics['recall'])
    avg_ndcg = np.mean(metrics['ndcg'])

    # plt.figure(figsize=(8, 5))
    # plt.bar(['Precision', 'Recall', 'NDCG'], [avg_precision, avg_recall, avg_ndcg], color='skyblue')
    # plt.ylim(0, 1)
    # plt.title(f"Evaluation Metrics (top@{k})")
    # plt.ylabel("Score")
    # plt.grid(True, axis='y')
    # plt.tight_layout()
    # plt.savefig("evaluation_metrics.png")
    # plt.show()

    return {
        f'precision@{k}': np.mean(metrics['precision']),
        f'recall@{k}': np.mean(metrics['recall']),
        f'ndcg@{k}': np.mean(metrics['ndcg']),
        'evaluation_time_sec': elapsed_time,
        'queries_evaluated': len(lines)
    }
    


def tune_hybrid_alpha(bm25_model, dense_model, queries_path, qrels_path, k=10):

    best_alpha = None
    best_score = -1
    results = []

    for alpha in np.linspace(0, 1, 11):  # Try alpha = 0.0 to 1.0
        #hybrid = HybridRetrievalModel(bm25_model, dense_model, alpha=alpha)
        hybrid = HybridReciprocalRankFusionModel(bm25_model, dense_model, alpha=alpha)
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

def grid_search(models_to_test, queries_path, qrels_path, texts, urls, faiss_index, faiss_mapping):
    best_results = []

    for model_type in models_to_test:
        if model_type == "bm25":
            for k1 in [1.2, 1.5, 1.8]:
                for b in [0.5, 0.75, 1.0]:
                    for expansion in [False, True]:
                        model = BM25RetrievalModel(texts, urls, k1=k1, b=b, use_expansion=expansion)
                        metrics = evaluate(model, queries_path, qrels_path, k=10)
                        best_results.append(("bm25", k1, b, expansion, metrics))

        elif model_type == "dense":
            for expansion in [False, True]:
                model = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=expansion)
                metrics = evaluate(model, queries_path, qrels_path, k=10)
                best_results.append(("dense", None, None, expansion, metrics))

        elif model_type == "hybrid_alpha":
            for alpha in np.linspace(0, 1, 11):
                for expansion in [False, True]:
                    bm25 = BM25RetrievalModel(texts, urls, use_expansion=expansion)
                    dense = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=expansion)
                    model = HybridAlphaModel(bm25, dense, alpha=alpha)
                    metrics = evaluate(model, queries_path, qrels_path, k=10)
                    best_results.append(("hybrid_alpha", alpha, None, expansion, metrics))

        elif model_type == "hybrid_rrf":
            for k_rrf in [20, 30, 40]:
                for expansion in [False, True]:
                    bm25 = BM25RetrievalModel(texts, urls, use_expansion=expansion)
                    dense = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=expansion)
                    model = HybridReciprocalRankFusionModel(bm25, dense, doc_mapping=faiss_mapping, k=k_rrf)
                    metrics = evaluate(model, queries_path, qrels_path, k=10)
                    best_results.append(("hybrid_rrf", k_rrf, None, expansion, metrics))

    return best_results

def plot_grid_comparison(results):

    labels = []
    ndcgs = []

    for model, param1, param2, expansion, metrics in results:
        label = f"{model}"
        if model == "bm25":
            label += f"\nk1={param1}, b={param2}, exp={expansion}"
        elif model == "dense":
            label += f"\nexp={expansion}"
        elif model == "hybrid_alpha":
            label += f"\nalpha={param1}, exp={expansion}"
        elif model == "hybrid_rrf":
            label += f"\nk={param1}, exp={expansion}"
        labels.append(label)
        ndcgs.append(metrics["ndcg@10"])

    plt.figure(figsize=(12, 6))
    plt.barh(labels, ndcgs, color='slateblue')
    plt.xlabel("NDCG@10")
    plt.title("Comparison of Models and Hyperparameters")
    plt.tight_layout()
    plt.savefig("grid_search_comparison.png")
    plt.show()

# --- TEST: Funktioniert das Dense-Modell? ---
with open("../indexing/indexing/output/doc_mapping.json", "r") as f:
    doc_mapping = json.load(f)
texts = [doc["title"] for doc in doc_mapping.values()]
urls = [doc["url"] for doc in doc_mapping.values()]
faiss_index, faiss_mapping = load_faiss_and_mapping(
    "../indexing/indexing/output/semantic_index.faiss",
    "../indexing/indexing/output/doc_mapping.json"
)

dense_test_model = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=False)
test_results = dense_test_model.retrieve("example query about AI", top_k=5)
print("\n🔍 Test: Dense Retrieval Output for 'example query about AI':")
for i, r in enumerate(test_results, 1):
    print(f"{i}. {r}")
print("\n-------------------------------------------------------------\n")
# --- ENDE TEST ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bm25", "dense", "hybrid_rrf", "hybrid_alpha"])
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha weight for BM25 in hybrid_alpha model")
    parser.add_argument("--use_expansion", action="store_true", help="Enable query expansion")
    parser.add_argument("--grid_search", action="store_true", help="Run full grid search across parameters")
    parser.add_argument("--use_reranker", action="store_true", help="Enable transformer reranking stage")

    


    args = parser.parse_args()


    # Load data for all models
    with open("../indexing/indexing/output/doc_mapping.json", "r") as f:
        doc_mapping = json.load(f)
    texts = [doc["title"] for doc in doc_mapping.values()]
    urls = [doc["url"] for doc in doc_mapping.values()]
    faiss_index, faiss_mapping = load_faiss_and_mapping(
        "../indexing/indexing/output/semantic_index.faiss",
        "../indexing/indexing/output/doc_mapping.json"
    )

    if args.grid_search:
        print("🔍 Starting grid search...")
        results = grid_search(["bm25", "dense", "hybrid_alpha", "hybrid_rrf"], args.queries, args.qrels, texts, urls, faiss_index, faiss_mapping)
        with open("grid_search_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("✅ Grid search results saved to grid_search_results.json")
        plot_grid_comparison(results)
    else:
        if not args.model:
            raise ValueError("You must provide --model unless --grid_search is specified")

        if args.model == "bm25":
            model = BM25RetrievalModel(texts, urls, use_expansion=args.use_expansion)
        elif args.model == "dense":
            model = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=args.use_expansion)
        elif args.model == "hybrid_rrf":
            bm25_model = BM25RetrievalModel(texts, urls, use_expansion=args.use_expansion)
            dense_model = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=args.use_expansion)
            model = HybridReciprocalRankFusionModel(bm25_model, dense_model, doc_mapping=faiss_mapping, k=20)
        elif args.model == "hybrid_alpha":
            bm25_model = BM25RetrievalModel(texts, urls, use_expansion=args.use_expansion)
            dense_model = DenseRetrievalModel(faiss_index, faiss_mapping, use_expansion=args.use_expansion)
            model = HybridAlphaModel(bm25_model, dense_model, alpha=args.alpha)
        else:
            raise ValueError(f"Unknown model: {args.model}")

    # Run evaluation
    metrics = evaluate(model, args.queries, args.qrels, k=args.topk)

    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")