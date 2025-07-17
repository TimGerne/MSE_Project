from hybrid_rrf import HybridReciprocalRankFusionModel
from models import BM25RetrievalModel, DenseRetrievalModel, load_faiss_and_mapping


# Load corpus (texts and urls)
import json
with open("../indexing/indexing/output/doc_mapping.json", "r") as f:
    doc_mapping = json.load(f)

texts = [doc["title"] for doc in doc_mapping.values()]
urls = [doc["url"] for doc in doc_mapping.values()]

# Load FAISS index and model
faiss_index, mapping = load_faiss_and_mapping(
    "../indexing/indexing/output/semantic_index.faiss",
    "../indexing/indexing/output/doc_mapping.json"
)

# Create base models
bm25_model = BM25RetrievalModel(texts, urls)
dense_model = DenseRetrievalModel(faiss_index, mapping)

# Create hybrid model
hybrid_model = HybridReciprocalRankFusionModel(bm25_model, dense_model, mapping)

# Test a query
query = "tübingen attractions"
results = hybrid_model.retrieve(query, top_k=10)

# Print results
for i, res in enumerate(results, 1):
    print(f"{i:2d}. {res['score']:.4f} | {res['title'][:60]} — {res['url']}")
