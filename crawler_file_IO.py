import os
import csv


def read_frontier():
    pass


def write_saved_pages(filename: str, pages_to_save: list) -> None:
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for page in pages_to_save:
            writer.writerow(page)


def save_frontier(filename: str, frontier: list) -> None:
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        for entry in frontier:
            writer.writerow(entry)


def empty_file(filename: str) -> None:
    cwd = os.getcwd()
    filepath = os.path.join(cwd, filename)
    if os.path.exists(filepath):
        # as we do not open in append mode we empty the file by writing nothing in it
        open(filename, 'w').close() 


