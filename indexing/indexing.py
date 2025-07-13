from lxml import html as lxml_html
import requests
import pandas as pd
import re
import json
import os
from language_detection import detect_language_from_html
from tokenize_utils import normalize_and_tokenize
from embedding_index import build_embeddings, build_faiss_index, save_doc_mapping, save_index
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple, Any, Optional


CSV_FILE: str = "../saved_pages.csv"
OUTPUT_DIR = "indexing/output/"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRAWLER_NAME: str = "MSE_Crawler_1"
TIMEOUT: int = 10
HEADERS: Dict[str, str] = {"User-Agent": CRAWLER_NAME}
INDEX_OUTPUT_FILE: str = OUTPUT_DIR + "semantic_index.faiss"
DOC_MAPPING_FILE: str = OUTPUT_DIR + "doc_mapping.json"
skipped_docs: List[Tuple[str, str]] = []



def fetch_pages(inputfile: str) -> pd.DataFrame:
    """
    Fetch HTML content from a list of URLs in a CSV file.

    Args:
        inputfile (str): Path to the CSV file containing URLs.

    Returns:
        pd.DataFrame: A DataFrame with columns: url, status, html, text, title, error (optional).
    """
    df_crawled_content = pd.read_csv(inputfile, header=None, encoding="latin1")
    urls = df_crawled_content.iloc[:, 2].dropna().astype(str)

    records = []
    for url in urls:
        try:
            response = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
            response.raise_for_status()

            html_text = response.text

            if detect_language_from_html(html_text) != "en":
                skipped_docs.append((url, 'not_english'))
                continue
            
            tree = lxml_html.fromstring(html_text)
            text = tree.text_content().strip()
            title_elem = tree.find(".//title")
            title = title_elem.text.strip() if title_elem is not None else ""

            records.append({
                "url": url,
                "status": response.status_code,
                "html": html_text,
                "text": text,
                "title": title
            })
        except Exception as e:
            records.append({
                "url": url,
                "status": None,
                "html": "",
                "text": "",
                "title": "",
                "error": str(e)
            })

    return pd.DataFrame(records)

def extract_metadata(tree: lxml_html.HtmlElement) -> Dict[str, Any]:
    """
    Extract structured metadata (JSON-LD and Open Graph) from an HTML tree.

    Args:
        tree (lxml_html.HtmlElement): Parsed HTML tree.

    Returns:
        Dict[str, Any]: Metadata dictionary with keys like 'json_ld' and 'open_graph'.
    """
    metadata = {}

    # Extract JSON-LD
    json_ld = []
    for script in tree.xpath('//script[@type="application/ld+json"]'):
        try:
            data = json.loads(script.text)
            json_ld.append(data)
        except Exception:
            pass
    if json_ld:
        metadata["json_ld"] = json_ld

    # Extract OpenGraph meta tags
    og_data = {}
    for meta in tree.xpath('//meta[starts-with(@property, "og:")]'):
        prop = meta.get("property", "")[3:]
        content = meta.get("content")
        if content:
            og_data[prop] = content
    if og_data:
        metadata["open_graph"] = og_data

    return metadata

def extract_useful_parts(html_string: str, min_par_len: int = 0) -> Dict[str, Any]:
    """
    Extract key content from HTML such as title, headings, and main body text.

    Args:
        html_string (str): Raw HTML content.
        min_par_len (int): Minimum length of paragraphs to keep.

    Returns:
        Dict[str, Any]: Dictionary with title, headings, main_content, metadata.
    """
    parser = lxml_html.HTMLParser(encoding="utf-8")
    tree = lxml_html.fromstring(html_string, parser=parser)

    # Remove noisy tags
    lxml_html.etree.strip_elements(tree, "script", "style", "noscript", "iframe", "nav", with_tail=False)

    # Title and headings
    title_elem = tree.find(".//title")
    title = title_elem.text.strip() if title_elem is not None else ""
    headings = [el.text_content().strip() for el in tree.xpath('//h1 | //h2 | //h3')]

    # Main content heuristic
    main_candidate = (tree.xpath('//article') or
                      tree.xpath('//main') or
                      tree.xpath('//body') or
                      [tree])[0]

    important_tags = ["p", "div", "li", "td", "th", "blockquote", "pre"]
    texts = []
    for tag in important_tags:
        for el in main_candidate.findall(f".//{tag}"):
            txt = el.text_content().strip()
            if len(txt) >= min_par_len:
                txt = re.sub(r"\s+", " ", txt)
                texts.append(txt)

    main_content = "\n\n".join(texts)
    metadata = extract_metadata(tree)

    return {
        "title": title,
        "headings": headings,
        "main_content": main_content,
        "metadata": metadata
    }


def build_index_dataframe(crawl_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw crawled data into structured content and tokens.

    Args:
        crawl_df (pd.DataFrame): Raw data with HTML content.

    Returns:
        pd.DataFrame: DataFrame with extracted text, title, headings, metadata, and tokens.
    """
    records = []
    for _, row in crawl_df.iterrows():
        html = row.get("html")
        if pd.isna(html) or not html:
            records.append({**row,
                "title": "",
                "headings": [],
                "main_content": "",
                "metadata": {},
                "tokens": []
            })
            continue

        extracted = extract_useful_parts(html)
        tokens = normalize_and_tokenize(extracted.get("main_content"))
        records.append({
                    **row,
                    **extracted,
                    "tokens": tokens
                })

    return pd.DataFrame(records)

def build_inverted_index_with_tf(df: pd.DataFrame) -> Dict[str, Dict[int, int]]:
    """
    Build an inverted index with term frequencies for each document.

    Args:
        df (pd.DataFrame): DataFrame with tokenized content.

    Returns:
        Dict[str, Dict[int, int]]: Inverted index with term frequencies.
    """
    inverted_index = defaultdict(dict)

    for doc_id, row in df.iterrows():
        tokens = row.get("tokens", [])
        term_freq = Counter(tokens)

        for term, freq in term_freq.items():
            inverted_index[term][doc_id] = freq

    return dict(inverted_index)


if __name__ == "__main__":
    # extract information
    df_pages = fetch_pages(CSV_FILE)
    df_index = build_index_dataframe(df_pages)
    
    # create document ids
    df_index = df_index.reset_index(drop=True)
    df_index["doc_id"] = df_index.index

    # Building token-based index
    index = build_inverted_index_with_tf(df_index)
    with open((OUTPUT_DIR + "inverted_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    # Building semantic index
    print("Creating sentence embeddings...")
    embeddings = build_embeddings(df_index)

    print("Building FAISS index...")
    faiss_index = build_faiss_index(embeddings)

    print("Saving FAISS index and document mapping...")
    save_index(faiss_index, INDEX_OUTPUT_FILE)
    save_doc_mapping(df_index, DOC_MAPPING_FILE)

    # Example for querying faiss index
    index = faiss.read_index(OUTPUT_DIR + "semantic_index.faiss")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    with open((OUTPUT_DIR + "doc_mapping.json"), "r", encoding="utf-8") as f:
        doc_mapping = json.load(f)

    query = "food and drinks"
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")

    k = 5  # top-k results
    D, I = index.search(query_vec, k)

    for i, dist in zip(I[0], D[0]):
        doc = doc_mapping[str(i)]
        print(f"({dist:.4f}) {doc['title']} — {doc['url']}")

    # pd.set_option("display.max_colwidth", 200)
    # print(df_index.loc[23, ["url"]])
    # row = df_index.loc[0, ["doc_id", "url", "title", "headings", "main_content", "metadata", "tokens"]]

