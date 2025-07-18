missing_urls_file = "missing_urls_report.txt"
original_qrels_file = "qrels.txt"
filtered_qrels_file = "qrels_filtered.txt"

# Load missing URLs
with open(missing_urls_file, "r", encoding="utf-8") as f:
    missing_urls = set(line.strip() for line in f if line.strip())

# Filter qrels
with open(original_qrels_file, "r", encoding="utf-8") as fin, \
     open(filtered_qrels_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        _, _, url, _ = parts
        if url not in missing_urls:
            fout.write(line)
