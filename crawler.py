import requests
from bs4 import BeautifulSoup
import time
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin
import heapq
from langdetect import detect_langs

CRAWLER_NAME = 'MSE_Crawler_1'
REQUEST_TIMEOUT = 10    # in seconds
TUEBINGENS = ['tübingen', 'tubingen', 'tuebingen']


visited = set()
parsers = {}

frontier = [(0, 'https://www.tuebingen.de/'),
            (0.5, 'https://docs.python.org/'),
            # this site blocks access by bots
            (1, 'https://www.tuebingen-info.de/'),
            (2, 'https://en.wikipedia.org/wiki/T%C3%BCbingen'),
            (3, 'https://en.wikipedia.org/wiki/T%C3%BCbingen#/media/File:Altstadt-tuebingen-1.jpg')]
heapq.heapify(frontier)


def parsing_allowed(url):
    # get domain
    domain = urlparse(url).netloc

    # check if parser exists and use or create it
    if domain in parsers:
        rp = parsers[domain]
    else:
        rp = RobotFileParser()
        rp.set_url(f"https://{domain}/robots.txt")
        parsers[domain] = rp

        # check if robots.tsx does exist and is readable
        try:
            rp.read()
        # if not allow to crawl
        except:
            return True

    # true if parsing allowed or not specified for our parser
    return rp.can_fetch(CRAWLER_NAME, url)


def get_crawl_delay(url, default_delay=1):
    # TODO make this more efficient as e.g. combine with parsing_allowed()
    domain = urlparse(url).netloc

    # domain should be in parser at this point as we called parsing_allowed before
    if domain in parsers:
        rp = parsers[domain]

    delay = rp.crawl_delay(CRAWLER_NAME)
    if delay:
        return delay
    else:
        return default_delay


# TODO was machen wir mit den dokumenten: - speichern? vektor repräsentationen? ...
# nur links speichern und dann die links von den embedding modellen aufrufen
def process_page(url, soup):
    try:
        print(soup.find('title').text)
        print(url)
    except:
        print(f'Page has no title')
        print(url)


def get_last_modified(response):
    # get the head of the response and check if it has Last-Modified tag
    last_modified = response.headers.get('Last-Modified')

    if last_modified:
        print(f'Last-Modified: {last_modified}')
    else:
        print('No Last-Modified information in page head')

    print()


def page_is_english(page_content, threshold=0.66) -> bool:
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


def is_unwanted_file_type(url) -> bool:
    unwanted_url_endings = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg",
                            ".webp", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx",
                            ".mp3", ".mp4", ".avi", ".mov", ".wmv", ".zip", ".rar", ".gz"]

    # TODO check if necessary and how long this takes
    unwanted_content_types = ["image/", "video/", "audio/", "application/pdf", "application/zip", "application/gzip",
                              "application/msword", "application/vnd.ms-excel", "application/vnd.ms-powerpoint",
                              "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              "application/vnd.openxmlformats-officedocument.presentationml.presentation"]

    # check for url endings
    for ending in unwanted_url_endings:
        if url.lower().endswith(ending):
            return True

    # only if url endings are ok, do request
    head_response = requests.head(url, headers={'User-Agent': CRAWLER_NAME},
                                  allow_redirects=True, timeout=REQUEST_TIMEOUT)
    page_content_type = head_response.headers.get('Content-Type', '').lower()

    for type in unwanted_content_types:
        if page_content_type.startswith(type):
            return True

    return False


def contains_tuebingen(text):
    for tue in TUEBINGENS:
        if tue in text.lower():
            return True

    return False


# crawls the frontier
def crawl():
    counter = len(frontier)

    while frontier:
        # get node with highest priority (i.e. lowest priority number)
        node = heapq.heappop(frontier)  # removes node from frontier
        url = node[1]
        priority = node[0]

        print(f'Priority: {priority}')

        if url in visited:
            print(f'URL has already been visited: {url}')
            continue

        visited.add(url)

        # get website
        response = requests.get(
            url, headers={'User-Agent': CRAWLER_NAME}, timeout=REQUEST_TIMEOUT)

        # check if website is available
        # we check this before checking robots.txt because if the used parser cannot access the website it does not throw an error
        if response.status_code != 200:
            print(f'URL returns wrong code: {url}')
            continue

        # check robots.tsx
        if not parsing_allowed(url):
            print(f'URL not allowed: {url}\n')
            continue

        time.sleep(get_crawl_delay(url))  # TODO when do we call this

        # we do not want to crawl images, powerpoint, ...
        if is_unwanted_file_type(url):
            print(f'Site is unwanted file type: {url}\n')

        # we only want to crawl english pages
        english_page, most_prob_lang = page_is_english(response.text)
        if not english_page:
            print(f'Site is not in english but {most_prob_lang}: {url}\n')
            continue

        # get page content
        soup = BeautifulSoup(response.text, 'html.parser')

        # do something with the content
        process_page(url, soup)
        get_last_modified(response)

        # check if current url or content contain word tübingen (in different spellings)
        tuebingen_url = contains_tuebingen(response.url)  # TODO decode url
        tuebingen_content = contains_tuebingen(response.text)
        # print(f'Tübingen URL: {tuebingen_url} and Tübingen content: {tuebingen_content}')

        # iterate through all links in page and add to priority queue
        for link in soup.find_all('a', href=True):

            # just returns link['href'] is the link is absolute
            next_url = urljoin(url, link['href'])

            if next_url.startswith('http') and next_url not in visited:
                heapq.heappush(frontier, (counter, next_url))
                counter += 1


def main():
    crawl()
    print(f'Number of visited sites: {len(visited)}')


if __name__ == '__main__':
    main()
