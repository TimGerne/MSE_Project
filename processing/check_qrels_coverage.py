import json
from collections import defaultdict

DOC_MAPPING_PATH = "../indexing/indexing/output/doc_mapping.json"
QRELS_PATH = "qrels.txt"
OUTPUT_MISSING_PATH = "missing_urls_report.txt"

# Load indexed URLs
with open(DOC_MAPPING_PATH, "r", encoding="utf-8") as f:
    doc_mapping = json.load(f)

indexed_urls = set(doc["url"] for doc in doc_mapping.values())

# Load qrels
qrels = defaultdict(list)
with open(QRELS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, url, rel = parts
        if int(rel) > 0:
            qrels[qid].append(url)

# Check coverage
total_qrels = 0
total_found = 0
missing_by_query = defaultdict(list)

for qid, urls in qrels.items():
    found = sum(1 for url in urls if url in indexed_urls)
    total_qrels += len(urls)
    total_found += found
    missing_by_query[qid] = [url for url in urls if url not in indexed_urls]
    print(f"{qid}: {found}/{len(urls)} qrel URLs are in the index")

coverage = total_found / total_qrels if total_qrels else 0
print(f"\nTotal coverage: {total_found}/{total_qrels} ({coverage:.2%})")

# Write missing URLs
with open(OUTPUT_MISSING_PATH, "w", encoding="utf-8") as f:
    for qid, urls in missing_by_query.items():
        if urls:
            f.write(f"{qid} missing:\n")
            for url in urls:
                f.write(f"  {url}\n")
            f.write("\n")

print(f"\nMissing URLs written to: {OUTPUT_MISSING_PATH}")
