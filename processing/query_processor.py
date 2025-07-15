import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os

# Paths to the index and document mapping
INDEX_PATH = "../indexing/output/semantic_index.faiss"
DOC_MAPPING_PATH = "../indexing/output/doc_mapping.json"

# Load SentenceTransformer model (same one used in indexing)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(INDEX_PATH)

# Load document metadata
with open(DOC_MAPPING_PATH, "r", encoding="utf-8") as f:
    doc_mapping = json.load(f)

def run_query(query: str, top_k: int = 100):
    """
    Runs a semantic query against the FAISS index and prints top results.
    """
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_vec, top_k)

    results = []
    for i, dist in zip(I[0], D[0]):
        doc = doc_mapping.get(str(i), {})
        title = doc.get("title", "")
        url = doc.get("url", "")
        results.append((dist, title, url))
    return results

def interactive_mode():
    print("Enter your query (or 'exit' to quit):")
    while True:
        query = input(">>> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        results = run_query(query)
        print_top_results(results)

def batch_mode(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                print(f"Skipping malformed line: {line}")
                continue
            query_id, query_text = parts
            print(f"\nQuery {query_id}: {query_text}")
            results = run_query(query_text)
            print_top_results(results)

def print_top_results(results):
    for rank, (dist, title, url) in enumerate(results, 1):
        print(f"{rank:3}. ({dist:.4f}) {title[:80]} — {url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search with SBERT and FAISS")
    parser.add_argument("--batch", type=str, help="Path to queries.txt for batch mode")
    args = parser.parse_args()

    if args.batch:
        if os.path.isfile(args.batch):
            batch_mode(args.batch)
        else:
            print(f"File not found: {args.batch}")
    else:
        interactive_mode()
