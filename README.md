# Tübingen Search Engine Project

This project implements a domain-specific search engine focused on retrieving **English-language web content about Tübingen**. It supports sparse (BM25), dense (semantic), and hybrid retrieval, with optional query expansion and neural reranking.

---

## Project Structure

```bash
.
├── crawling/   
    ├── crawler.py                  # Main script for crawling
    ├── crawler_file_IO.py          # External file management
    ├── frontier_seeds.txt          # Starting point for new crawl
├── indexing/  
    ├── embedding_index.py          # Creation of Embedding and FAISS index 
    ├── indexing.py                 # Main script to create indeces
    ├── language_detection.py       # Functionality to detect language of webpage
    ├── tokenize_utils.py           # Content Preprocessing 
    ├── output
        ├── semantic_index.faiss    # Currently used semantic Index
        ├── inverted_index.json     # Token-based Index
        ├── doc_mapping.json        # Mapping Document to ID
├── processing/             
    ├── models.py                   # BM25, Dense, HybridReciprocalRankFusionModel and HybridAlphaModel  
    ├── reranker.py                 # Optional reranker (cross-encoder)
    ├── evaluation.py               # Main script to evaluate models
    ├── queries.txt                 # Evaluation queries
    ├── qrels_filtered.txt          # Relevance judgments
    ├── grid_search_results.json    # Output files (if generated)
    ├── retrieval_interface.py      # takes a list of queries and returns a Pandas DataFrame
└── README.md               
```

---

## How to Run

Run UI:

```bash
streamlit run .processing/UI.py
```


Run evaluation using different models:

```bash
# Hybrid RRF without reranking
python evaluation.py --model hybrid_rrf --queries queries.txt --qrels qrels_filtered.txt

# With reranking and/or query expansion
python evaluation.py --model hybrid_rrf --queries queries.txt --qrels qrels_filtered.txt --use_reranker --use_expansion
```

Dependencies:

```bash
pip install -r requirements.txt
```

---

## Models Implemented

* **BM25RetrievalModel**
  Classic lexical model based on tokenized terms and term frequency.

* **DenseRetrievalModel**
  Vector-based semantic search using `all-MiniLM-L6-v2` from SentenceTransformers and FAISS.

* **HybridAlphaModel**
  Linear interpolation of BM25 and Dense scores. Controlled by an `--alpha` parameter.

* **HybridReciprocalRankFusionModel (RRF)**
  Final model used. Combines rankings from BM25 and Dense using reciprocal rank fusion.

---

## Query Processing Pipeline

### Tokenization & Normalization

* Lowercasing, punctuation removal, whitespace normalization
* Ensures compatibility with indexed document tokens

### Query Expansion (tested)

Three expansion strategies were implemented:

1. **WordNet Synonyms**
   Expanded query terms using WordNet. Introduced noise in most cases.

2. **Pseudo-Relevance Feedback (PRF)**
   Added frequent terms from top-ranked BM25 results. Slight recall boost, but harmed precision.

3. **GloVe-based Semantic Expansion**
   Expanded terms via nearest neighbors in GloVe 300d space, filtered by corpus vocabulary.

**Outcome:** All methods slightly reduced NDCG and precision. Final system **disables query expansion by default**.

### Neural Reranking (tested)

* Used cross-encoder (`stsb-TinyBERT` or `cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank top-100 results
* Re-encoded (query, doc) pairs for more accurate scoring
* **Drawbacks:** Increased eval time \~80x; no consistent improvement in ranking
* **Decision:** Disabled in final evaluation but kept as optional module (`--use_reranker`)

---

## Evaluation Results (Top-10)


---

## Final Notes

* **Final model**: `HybridReciprocalRankFusionModel` with `k=20`, no expansion, no reranker
* Query expansion and reranking were tested extensively but excluded due to reduced effectiveness or runtime issues
* Codebase is modular and allows plug-and-play of components for future testing

---

## Author

Aline Breitinger, Moritz Christ, Lili Even, Tim Gerne und Jonathan Nemitz

 – Modern Search Engines Project – Summer Term 2025
