import os
import csv


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


def save_visited(filename: str, visited: set) -> None:
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for url in visited:
            writer.writerow([url])


def empty_file(filename: str) -> None:
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    if os.path.exists(filepath):
        # as we do not open in append mode we empty the file by writing nothing in it
        open(filename, 'w').close()
    print('Emptying done')


def read_saved_frontier(filename: str) -> list:
    result = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Convert first two columns to int, third remains a string
            tup = (int(row[0]), int(row[1]), row[2])
            result.append(tup)
    return result


def read_saved_visited(filename: str) -> list:
    result = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result.extend(row)
    return result
