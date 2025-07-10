import requests
from bs4 import BeautifulSoup
import time
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin, unquote_plus
import heapq
from langdetect import detect_langs

from crawler_file_IO import write_saved_pages, save_frontier, save_visited, empty_file, read_saved_frontier, read_saved_visited


# set this to False if you want to start the crawl with a given frontier, and visited set
START_NEW_SEARCH = False
# after how many crawler loop iterations frontier and visited_pages get saved to file
CHUNKSIZE = 10  # TODO set bigger value for deployment

CRAWLER_NAME = 'MSE_Crawler_1'
REQUEST_TIMEOUT = 10    # in seconds
TUEBINGENS = ['tübingen', 'tubingen', 'tuebingen']


parsers = {}

# entry has pattern (priority_score, depth, url)
default_frontier = [(-1000, 0, 'https://www.tuebingen.de/'),
                    # this site blocks access by bots
                    (-999, 0, 'https://www.tuebingen-info.de/'),
                    (-998, 0, 'https://en.wikipedia.org/wiki/T%C3%BCbingen'),
                    (-997, 0, 'https://www.reddit.com/r/Tuebingen/')]


def parsing_allowed(url: str) -> bool:
    # get domain
    domain = urlparse(url).netloc

    # check if parser exists and use or create it
    if domain in parsers:
        rp = parsers[domain]
    else:
        try:
            rp = RobotFileParser()
            rp.set_url(f"https://{domain}/robots.txt")
            parsers[domain] = rp

            # check if robots.tsx does exist and is readable
            rp.read()
            # if not allow to crawl
        except Exception as e:
            print(f"[ERROR] Failed to read robots.txt for {domain}: {e}")
            return True

    # true if parsing allowed or not specified for our parser
    return rp.can_fetch(CRAWLER_NAME, url)


# has to be called after parsing_allowed to make sure that domain is in parsers dic
def get_crawl_delay(url: str, default_delay: int = 1) -> int:
    # TODO make this more efficient as e.g. combine with parsing_allowed()
    domain = urlparse(url).netloc

    # domain should be in parser at this point as we called parsing_allowed before
    try:
        rp = parsers[domain]

        delay = rp.crawl_delay(CRAWLER_NAME)
        if delay:
            return delay
        else:
            return default_delay
    except Exception as e:
        print(f"[ERROR] getting delay failed for {url}: {e}")
        return default_delay


# TODO was machen wir mit den dokumenten: - speichern? vektor repräsentationen? ...
# nur links speichern und dann die links von den embedding modellen aufrufen
def process_page(url: str, soup: BeautifulSoup) -> None:
    try:
        print(soup.find('title').text)
        print(url)
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

# TODO threshold value


def page_is_english(page_content, threshold: int = 0.66) -> bool:
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
        return True  # TODO might change this to False


def is_unwanted_file_type(url: str) -> bool:
    unwanted_url_endings = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg",
                            ".webp", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
                            ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".zip", ".rar", ".gz"]

    # TODO check if necessary and how long this takes
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
        return True  # TODO might change this to False


def contains_tuebingen(text: str) -> bool:
    try:
        for tue in TUEBINGENS:
            if tue in text.lower():
                return True

        return False

    except Exception as e:
        print(f'[ERROR] checking if Tuebingen in page: {e}')
        return False


def calc_priority_score(url: str, depth: int, anchor_text: str) -> int:
    # best score is 27
    score = 0

    score += max(0, 15 - depth)

    if contains_tuebingen(unquote_plus(url)):
        score += 5

    if contains_tuebingen(anchor_text):
        score += 7

    return score


# crawls the frontier
def crawl(frontier, visited):
    n_iterations = 0
    pages_to_save = []              # keeps track of visited pages
    # keeps track of sites the crawler was not allowed to visit
    blocking_pages_to_save = []

    while frontier:
        if n_iterations % CHUNKSIZE == 0:
            save_frontier('frontier.csv', frontier)

            if pages_to_save:
                write_saved_pages('saved_pages.csv', pages_to_save)
                pages_to_save = []  # empty the list

            if blocking_pages_to_save:
                write_saved_pages('blocked_saved_pages.csv',
                                  blocking_pages_to_save)
                blocking_pages_to_save = []  # empty the list

            save_visited('visited.csv', visited)

        # get node with highest priority (i.e. lowest priority number)
        node = heapq.heappop(frontier)  # removes node from frontier
        priority_score = node[0]
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
        # currently we save even if we cannot open the page TODO think about this
        # pages_to_save.append(node)

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
        # we check this before checking robots.txt because if the used parser cannot access the website it does not throw an error
        if response.status_code != 200:
            print(f'URL returns wrong code: {url}')
            continue

        # we only want to crawl english pages
        english_page, most_prob_lang = page_is_english(response.text)
        if not english_page:
            print(f'Site is not in english but {most_prob_lang}: {url}\n')
            continue

        time.sleep(get_crawl_delay(url))  # TODO when do we call this

        # get page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # do something with the content
        process_page(url, soup)
        # get_last_modified(response)
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

        frontier = default_frontier
        visited = set()

    else:
        frontier = read_saved_frontier('frontier.csv')
        heapq.heapify(frontier)

        visited = set(read_saved_visited('visited.csv'))

    heapq.heapify(frontier)
    crawl(frontier, visited)


if __name__ == '__main__':
    main()
