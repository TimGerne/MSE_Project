import requests
from bs4 import BeautifulSoup
import time
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin, unquote_plus
import heapq
from langdetect import detect_langs
from simhash import Simhash
import psutil

from crawler_file_IO import (write_saved_pages, save_frontier, save_set_to_csv, empty_file,
                             read_saved_frontier, read_saved_visited, read_saved_hashes, read_frontier_seeds, count_entries_in_csv)


# TODO set this to False if you want to start the crawl with a given frontier, and visited set
START_NEW_SEARCH = False
# TODO set to specify after how many crawler loops information is saved
CHUNKSIZE = 10

CRAWLER_NAME = 'MSE_Crawler_1'
REQUEST_TIMEOUT = 10    # in seconds
TUEBINGENS = ['tübingen', 'tubingen', 'tuebingen']
SIMHASH_THRESHOLD = 5

parsers = {}


def parsing_allowed(url: str) -> bool:
    # get domain
    domain = urlparse(url).netloc

    # check if parser exists and use it or create it
    if domain in parsers:
        rp = parsers[domain]
    else:
        try:
            rp = RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            parsers[domain] = rp

            # check if robots.tsx does exist and is readable
            rp.read()

            # This is an edge case were we cannot access robots.txt (e.g. because of rerouting to main page)
            # We assume that we are allowed to crawl in this case
            if not rp.last_checked:
                return True
        except Exception as e:
            print(f"[ERROR] Failed to read robots.txt for {domain}: {e}")
            return True

    # true if parsing allowed or not specified for our parser
    return rp.can_fetch(CRAWLER_NAME, url)


# has to be called after parsing_allowed to make sure that domain is in parsers dic
def get_crawl_delay(url: str, default_delay: float = 1.0) -> float:
    domain = urlparse(url).netloc

    # domain should be in parser at this point as we called parsing_allowed before
    try:
        rp = parsers[domain]

        delay = rp.crawl_delay(CRAWLER_NAME)
        if delay and delay <= 10:
            return delay
        else:
            return default_delay
    except Exception as e:
        print(f"[ERROR] getting delay failed for {url}: {e}")
        return default_delay


def process_page(url: str, soup: BeautifulSoup) -> None:
    try:
        print(soup.find('title').text)
        print(url)
        print()
    except Exception as e:
        print(f'[ERROR] Page title could not be read: {e}')
        print(url)


def get_last_modified(response: requests.Response) -> None:
    try:
        # get the head of the response and check if it has Last-Modified tag
        last_modified = response.headers.get('Last-Modified')

        if last_modified:
            print(f'Last-Modified: {last_modified}')
        else:
            print('No Last-Modified information in page head')

        print()
    except Exception as e:
        print(f'[ERROR] while retrieving page last modified data: {e}')


def page_is_english(page_content, threshold: int = 0.66) -> tuple[bool, str]:
    try:
        # parse website again (needed as changes to soup are permanent)
        soup = BeautifulSoup(page_content, 'html.parser')

        # remove html tags
        for tag in soup(['script', 'style']):
            tag.decompose()

        # just get readable text
        text = soup.get_text(separator=' ', strip=True)

        # create dictionary with languages and their probabilities, based on naive bayes
        # which/how many languages should be included cannot be controlled
        lang_probs = {item.lang: item.prob for item in detect_langs(text)}
        prob_english = lang_probs.get('en', 0)

        most_prob_lang = next(iter(lang_probs))

        return prob_english >= threshold, most_prob_lang

    except Exception as e:
        print(f'[ERROR] while checking page language: {e}')
        return True  # assume page is english when we cannot get its language


def is_unwanted_file_type(url: str) -> bool:
    unwanted_url_endings = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg",
                            ".webp", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
                            ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".zip", ".rar", ".gz"]

    unwanted_content_types = ["image/", "video/", "audio/", "application/pdf", "application/zip", "application/gzip",
                              "application/msword", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
                              "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              "application/vnd.openxmlformats-officedocument.presentationml.presentation"]

    try:
        # check for url endings
        for ending in unwanted_url_endings:
            if url.lower().endswith(ending):
                return True

        # only if url endings are ok, do request
        head_response = requests.head(url, headers={'User-Agent': CRAWLER_NAME},
                                      allow_redirects=True, timeout=REQUEST_TIMEOUT)
        page_content_type = head_response.headers.get(
            'Content-Type', '').lower()

        for type in unwanted_content_types:
            if page_content_type.startswith(type):
                return True

        return False

    except Exception as e:
        print(f'[ERROR] while checking for unwanted file type: {e}')
        return True     # we are cautious here if we cannot check the filetype


def contains_tuebingen(text: str) -> bool:
    try:
        for tue in TUEBINGENS:
            if tue in text.lower():
                return True

        return False

    except Exception as e:
        print(f'[ERROR] checking if Tuebingen in page: {e}')
        return False


def check_duplicate(soup, all_hashes: set, threshold: int = 3) -> tuple[bool, set]:
    # remove unwanted tags
    for tag in soup(['script', 'style']):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    tokens = text.lower().split()
    curr_hash = Simhash(tokens).value

    for hash in all_hashes:
        hamming_distance = bin(curr_hash ^ hash).count('1')
        if hamming_distance <= threshold:
            return True, all_hashes

    # only add when no duplicate is found, as otherwise we skip the page
    all_hashes.add(curr_hash)

    return False, all_hashes


def calc_priority_score(url: str, depth: int, anchor_text: str) -> int:
    # best score is 27
    score = 0

    score += max(0, 15 - depth)

    if contains_tuebingen(unquote_plus(url)):
        score += 5

    if contains_tuebingen(anchor_text):
        score += 7

    return score


def save_files(frontier, visited, all_hashes, pages_to_save, blocking_pages_to_save):
    save_frontier('frontier.csv', frontier)
    save_set_to_csv('visited.csv', visited)
    save_set_to_csv('all_hashes.csv', all_hashes)

    if pages_to_save:
        write_saved_pages('saved_pages.csv', pages_to_save)
        pages_to_save = []  # empty list as its just used for this file

    if blocking_pages_to_save:
        write_saved_pages('blocked_saved_pages.csv',
                          blocking_pages_to_save)
        blocking_pages_to_save = []  # empty list as its just used for this file

    len_frontier = count_entries_in_csv('frontier.csv')
    len_saved_pages = count_entries_in_csv('saved_pages.csv')
    print(
        f'\nAmount of saved pages: {len_saved_pages} | Frontier size: {len_frontier}\n')

    return pages_to_save, blocking_pages_to_save


def crawl(frontier, visited: set, all_hashes: set) -> None:
    n_iterations = 0
    # keeps track of visited pages
    pages_to_save = []
    # keeps track of sites the crawler was not allowed to visit
    blocking_pages_to_save = []

    while frontier:
        # makes sure RAM does not get full
        if psutil.virtual_memory().percent > 90:
            pages_to_save, blocking_pages_to_save = save_files(
                frontier, visited, all_hashes, pages_to_save, blocking_pages_to_save)
            print("[WARNING] Memory usage high. Saving and stopping crawler")
            quit()

        if n_iterations % CHUNKSIZE == 0:
            pages_to_save, blocking_pages_to_save = save_files(
                frontier, visited, all_hashes, pages_to_save, blocking_pages_to_save)

        # get node with highest priority (i.e. lowest priority number)
        node = heapq.heappop(frontier)  # get and remove node from frontier
        priority_score = node[0]
        # how many "steps" this node is away from frontier seeds
        depth = node[1]
        url = node[2]

        n_iterations += 1
        print(f'n={n_iterations} | Priority: {priority_score} | Depth: {depth}')

        # we do not want to crawl images, powerpoint, ...
        if is_unwanted_file_type(url):
            print(f'Site is unwanted file type: {url}\n')
            continue

        if url in visited:
            print(f'URL has already been visited: {url}\n')
            continue

        visited.add(url)

        # check robots.tsx
        if not parsing_allowed(url):
            print(f'URL not allowed: {url}\n')
            blocking_pages_to_save.append(node)
            continue

        # get website
        try:
            response = requests.get(
                url, headers={'User-Agent': CRAWLER_NAME}, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            print(f'[ERROR] while fetching url {url}: {e}')
            continue

        # check if website is available
        if response.status_code != 200:
            print(f'URL returns wrong code {response.status_code}: {url}\n')
            continue

        # we only want to crawl english pages
        english_page, most_prob_lang = page_is_english(response.text)
        if not english_page:
            print(f'Site is not in english but {most_prob_lang}: {url}\n')
            continue

        # get page content
        soup = BeautifulSoup(response.text, 'html.parser')

        is_duplicate, all_hashes = check_duplicate(soup, all_hashes)
        if is_duplicate:
            print(f'Page is duplicate: {url}')
            continue

        time.sleep(get_crawl_delay(url))

        # do something with the content
        process_page(url, soup)
        pages_to_save.append(node)

        # iterate through all links in page and add to priority queue
        for link in soup.find_all('a', href=True):

            # just returns link['href'] is the link is absolute
            next_url = urljoin(url, link['href'])

            if next_url.startswith('http') and next_url not in visited:
                anchor_text = link.get_text(strip=True)
                # negate score as we use a min heap
                score = - calc_priority_score(next_url, depth, anchor_text)

                # depth used as secondary priority counter when two entries have same score
                heapq.heappush(frontier, (score, (depth+1), next_url))


def main():
    if START_NEW_SEARCH:
        for file in ['frontier.csv', 'saved_pages.csv', 'blocked_saved_pages.csv', 'visited.csv']:
            empty_file(file)
        print()

        frontier = read_frontier_seeds('frontier_seeds.txt')
        visited = set()
        all_hashes = set()

    else:
        frontier = read_saved_frontier('frontier.csv')
        visited = set(read_saved_visited('visited.csv'))
        all_hashes = set(read_saved_hashes('all_hashes.csv'))

    heapq.heapify(frontier)
    crawl(frontier, visited, all_hashes)


if __name__ == '__main__':
    main()
