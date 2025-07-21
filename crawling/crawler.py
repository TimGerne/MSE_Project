import requests
import os
import time
import heapq
import psutil
import sys
import random
from typing import List, Set, Dict, Tuple
from bs4 import BeautifulSoup
from langdetect import detect_langs
from simhash import Simhash
from urllib.parse import urlparse, unquote_plus
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin, unquote_plus


from crawler_file_IO import (write_saved_pages, save_frontier, save_set_to_csv, empty_file,
                             read_saved_frontier, read_saved_visited, read_saved_hashes,
                             read_frontier_seeds, count_entries_in_csv, save_domain_counts, read_domain_counts)


# Set to True to start a fresh crawl from a given frontier, clearing all saved files.
# Set to False to resume the crawl from saved files.
START_NEW_SEARCH = False
# Set to specify after how many crawler loops information is saved
CHUNKSIZE = 50

CRAWLER_NAME = 'MSE_Crawler_1'
REQUEST_TIMEOUT = 10    # in seconds
TUEBINGENS = ['tübingen', 'tubingen', 'tuebingen']
SIMHASH_THRESHOLD = 50

parsers = {}


def parsing_allowed(url: str) -> bool:
    """
    Check whether a given URL is allowed to be crawled based on robots.txt rules.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if crawling is allowed or robots.txt is unavailable; False if disallowed.
    """

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


def get_crawl_delay(url: str, default_delay: float = 1.0) -> float:
    """
    Retrieve the crawl delay for a given URL based on the site's robots.txt rules.

    Args:
        url (str): The URL to check for crawl delay.
        default_delay (float, optional): Default delay to return if not specified or on error. Defaults to 1.0.

    Returns:
        float: Crawl delay in seconds, either specified in robots.txt or the default.
    """

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
    """
    Process a webpage by printing its title and URL.

    Args:
        url (str): The URL of the page.
        soup (BeautifulSoup): Parsed HTML content of the page.

    Returns:
        None
    """

    try:
        print(soup.find('title').text)
        print(url)
        print()
    except Exception as e:
        print(f'[ERROR] Page title could not be read: {e}')
        print(url)


def page_is_english(page_content: str, threshold: int = 0.66) -> tuple[bool, str]:
    """
    Determine whether the content of a webpage is in English based on language detection.

    Args:
        page_content (str): Raw HTML content of the page.
        threshold (int, optional): Minimum probability required to consider the page English. Defaults to 0.66.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating if the page is in English,
                          and the most probable detected language code.
    """

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
        # assume page is english when we cannot get its language
        return True, 'language not found'


def is_unwanted_file_type(url: str) -> bool:
    """
    Determine whether a URL points to an unwanted file type based on its extension or subdomain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is likely to point to a non-HTML file (e.g., image, video, document); False otherwise.
    """

    unwanted_url_endings = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg",
                            ".webp", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
                            ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".zip", ".rar", ".gz", ".faces"]

    unwanted_url_beginnings = ["image.", "video.", "audio."]
    try:
        # check for url endings
        for ending in unwanted_url_endings:
            if url.lower().endswith(ending):
                return True

        # this is a bit of a  heuristic but seems to work
        for content_type in unwanted_url_beginnings:
            if urlparse(url).netloc.startswith(content_type):
                return True

        return False

    except Exception as e:
        print(f'[ERROR] while checking for unwanted file type: {e}')
        return True     # we are cautious here if we cannot check the filetype


def contains_tuebingen(text: str) -> bool:
    """
    Check whether a given text contains any variation of "Tübingen".

    Args:
        text (str): The text to search.

    Returns:
        bool: True if any known variation of "Tübingen" is found in the text; False otherwise.
    """

    try:
        for tue in TUEBINGENS:
            if tue in text.lower():
                return True

        return False

    except Exception as e:
        print(f'[ERROR] checking if Tuebingen in page: {e}')
        return False


def check_duplicate(soup: BeautifulSoup, all_hashes: Set, threshold: int = 3) -> tuple[bool, set]:
    """
    Check whether a webpage is a near-duplicate based on SimHash similarity.

    Args:
        soup (BeautifulSoup): Parsed HTML content of the page.
        all_hashes (set): A set of SimHash values for previously seen pages.
        threshold (int, optional): Maximum Hamming distance to consider pages as duplicates. Defaults to 3.

    Returns:
        tuple[bool, set]: A tuple where the first value indicates if the page is a near-duplicate,
                          and the second is the updated set of hashes.
    """

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


def calc_priority_score(url: str, depth: int, anchor_text: str) -> float:
    """
    Calculate a priority score for a URL based on crawl depth and presence of target keyword Tübingen.

    Args:
        url (str): The URL to score.
        depth (int): The crawl depth of the URL.
        anchor_text (str): The anchor text associated with the URL.

    Returns:
        float: A priority score where higher values indicate higher priority.
    """

    # best score is 27
    score = 0

    score += max(0, 15 - depth)

    if contains_tuebingen(unquote_plus(url)):
        score += 5

    if contains_tuebingen(anchor_text):
        score += 7

    return float(score)

# This was used to see whether we can get a more diverse set of save pages
# More information can be found in project report


def calc_priority_score_updated(url: str, depth: int, anchor_text: str, parent_score: float,
                                domain_counts: Dict, max_depth: int = 10) -> float:
    """
    Calculate a priority score for a URL based on depth, keywords, domain diversity,
    parent score, and exploration randomness.

    Args:
        url (str): The URL to score.
        depth (int): The crawl depth of the URL.
        anchor_text (str): The anchor text associated with the URL.
        parent_score (float): The priority score of the parent URL.
        domain_counts (dict): A dictionary tracking visit counts per domain.
        max_depth (int, optional): Maximum crawl depth allowed. Defaults to 10.

    Returns:
        float: The calculated priority score, with higher scores indicating higher priority.
    """

    score = 0.0
    current_domain = urlparse(url).netloc

    # 1. Soft penalty for going too deep
    if depth > max_depth:
        print(f"[WARNING] Max depth exceeded for {url}")
        return 1  # very low priority

    # 2. Depth bonus
    score += max(0, 12 - depth)

    # 3. Keyword in URL (Tuebingen)
    if contains_tuebingen(unquote_plus(url)):
        score += 4

    # 4. Keyword in anchor text
    if contains_tuebingen(anchor_text):
        score += 4

    # 5. Domain diversity bonus using smooth decay
    domain_hits = domain_counts.get(current_domain, 0)
    domain_bonus = min(5, 5 / (1 + domain_hits))
    score += domain_bonus
    domain_counts[current_domain] = domain_hits + 1

    # 6. Inherit 20% of parent score
    score += abs(parent_score * 0.2)

    # 7. Random exploration scaled with depth
    exploration_bonus = random.randint(0, int(2 * (1 - depth / max_depth)))
    score += exploration_bonus

    return score


def save_files(frontier: List[Tuple[float, int, str]], visited: Set[str], all_hashes: Set[int],
               pages_to_save: List[Tuple[float, int, str]], blocking_pages_to_save: List[Tuple[float, int, str]],
               domain_counts: Dict[str, int]) -> Tuple[List[Tuple[float, int, str]], List[Tuple[float, int, str]]]:
    """
    Save the crawler's current state and data to various CSV files, including the frontier, visited URLs, content hashes,
    domain visit counts, and saved pages. After saving, resets the lists of pages to save.

    Args:
        frontier (list[tuple[float, int, str]]): Priority queue of URLs to crawl, each as (score, depth, url).
        visited (set[str]): Set of URLs that have already been visited.
        all_hashes (set[int]): Set of content hashes used to detect duplicate pages.
        pages_to_save (list[tuple[float, int, str]]): List of visited pages to be saved.
        blocking_pages_to_save (list[tuple[float, int, str]]): List of pages that blocked access.
        domain_counts (dict[str, int]): Dictionary tracking the count of URLs per domain.

    Returns:
        tuple[list[tuple[float, int, str]], list[tuple[float, int, str]]]:
            Updated (cleared) lists for `pages_to_save` and `blocking_pages_to_save` after saving.
    """

    save_frontier('frontier.csv', frontier)

    save_set_to_csv('visited.csv', visited)
    save_set_to_csv('all_hashes.csv', all_hashes)
    save_domain_counts('domain_counts.csv', domain_counts)

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


def crawl(frontier, visited: Set, all_hashes: Set, domain_counts: Dict) -> None:
    """
    Perform a prioritized web crawl starting from the given frontier, managing visited URLs, duplicates, and domain counts.

    The crawler processes URLs in order of priority, respects robots.txt rules, filters unwanted file types,
    skips non-English or duplicate pages, and saves its state periodically or when memory is high.

    Args:
        frontier (list[tuple[float, int, str]]): Priority queue (min-heap) of URLs to crawl, each tuple contains
            (priority_score, depth, url).
        visited (set[str]): Set of URLs that have already been visited.
        all_hashes (set[int]): Set of content hashes to detect duplicate pages.
        domain_counts (dict[str, int]): Counts of how many URLs have been visited per domain.

    Returns:
        None
    """

    n_iterations = 0
    # keeps track of visited pages
    pages_to_save = []
    # keeps track of sites the crawler was not allowed to visit
    blocking_pages_to_save = []

    while frontier:
        # makes sure RAM does not get full
        if psutil.virtual_memory().percent > 90:
            pages_to_save, blocking_pages_to_save = save_files(
                frontier, visited, all_hashes, pages_to_save, blocking_pages_to_save, domain_counts)
            print("[WARNING] Memory usage high. Saving and stopping crawler")
            sys.exit()

        if n_iterations % CHUNKSIZE == 0:
            pages_to_save, blocking_pages_to_save = save_files(
                frontier, visited, all_hashes, pages_to_save, blocking_pages_to_save, domain_counts)

        # get node with highest priority (i.e. lowest priority number)
        node = heapq.heappop(frontier)  # get and remove node from frontier
        priority_score = node[0]
        # how many "steps" this node is away from frontier seeds
        depth = node[1]
        url = node[2]

        n_iterations += 1
        print(f'n={n_iterations} | Priority: {priority_score} | Depth: {depth}')
        print(url)

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
            print(f'Page is duplicate: {url}\n')
            continue

        time.sleep(get_crawl_delay(url))

        # do something with the content
        process_page(url, soup)
        pages_to_save.append(node)

        # iterate through all links in page and add to priority queue
        for link in soup.find_all('a', href=True):

            # some mailing links are not filtered out properly by the parsers otherwise
            if '[at]' in link['href'].lower():
                continue

            # just returns link['href'] is the link is absolute
            next_url = urljoin(url, link['href'])

            if next_url.startswith('http') and next_url not in visited:
                anchor_text = link.get_text(strip=True)
                # negate score as we use a min heap

                score = - calc_priority_score(url, depth, anchor_text)
                # score = - calc_priority_score_updated(url, depth, anchor_text, priority_score, domain_counts)

                # depth used as secondary priority counter when two entries have same score
                heapq.heappush(frontier, (score, depth+1, next_url))


def main():
    """
    Initialize and run the web crawler.

    Depending on the START_NEW_SEARCH flag, either starts a fresh crawl by
    clearing previous data files and loading seed URLs, or resumes from saved
    crawl state by loading frontier, visited URLs, hashes, and domain counts.

    Args:
        None

    Returns:
        None
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if START_NEW_SEARCH:
        for file in ['frontier.csv', 'saved_pages.csv', 'blocked_saved_pages.csv', 'visited.csv']:
            empty_file(file)
        print()

        frontier = read_frontier_seeds('frontier_seeds.txt')
        visited = set()
        all_hashes = set()
        domain_counts = dict()

    else:
        frontier = read_saved_frontier('frontier.csv')
        # frontier = read_saved_frontier('new_prioritized_frontier.csv')
        visited = set(read_saved_visited('visited.csv'))
        all_hashes = set(read_saved_hashes('all_hashes.csv'))

        domain_counts = read_domain_counts('domain_counts.csv')

    heapq.heapify(frontier)
    crawl(frontier, visited, all_hashes, domain_counts)


if __name__ == '__main__':
    main()
