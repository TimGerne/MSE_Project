import os
import csv


def empty_file(filename: str) -> None:
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    if os.path.exists(filepath):
        # as we do not open in append mode we empty the file by writing nothing in it
        open(filename, 'w').close()
    print(f'Emptied {filename}')


def count_entries_in_csv(file_path: str) -> int:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0


def write_saved_pages(filename: str, pages_to_save: list) -> None:
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for page in pages_to_save:
            writer.writerow(page)


def save_frontier(filename: str, frontier: list) -> None:
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for entry in frontier:
            writer.writerow(entry)


def save_set_to_csv(filename: str, visited: set) -> None:
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for url in visited:
            writer.writerow([url])


def save_domain_counts(filename: str, domain_counts: dict):
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["domain", "count"])
        for domain, count in domain_counts.items():
            writer.writerow([domain, count])


def read_saved_frontier(filename: str) -> list:
    result = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Convert first two columns to int, third remains a string
            tup = (float(row[0]), int(row[1]), row[2])
            result.append(tup)
    return result


def read_saved_visited(filename: str) -> list:
    result = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result.extend(row)
    return result


def read_saved_hashes(filename: str) -> list:
    result = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result.append(int(row[0]))
    return result


def read_frontier_seeds(filepath: str, start_priority: int = -1000, start_depth: int = 0) -> list:
    default_frontier = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, url in enumerate(lines):
        default_frontier.append((start_priority + i, start_depth, url))

    return default_frontier


def read_domain_counts(filename: str = "domain_counts.csv") -> dict:
    domain_counts = {}
    try:
        with open(filename, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                domain = row["domain"]
                count = int(row["count"])
                domain_counts[domain] = count
    except FileNotFoundError:
        print(f"[INFO] No previous domain count file found: {filename}")
    return domain_counts
