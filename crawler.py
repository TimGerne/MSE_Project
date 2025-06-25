import requests
from bs4 import BeautifulSoup
import time
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin

CRAWLER_NAME = 'MSE_Crawler_1'

seeds = ['https://www.wikipedia.org']
visited = set()
parsers = {}


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
    else:
        print(parsers)
        raise ('TEST')

    delay = rp.crawl_delay(CRAWLER_NAME)
    if delay:
        return delay
    else:
        return default_delay


# TODO
def process_page(url, soup):
    try:
        print(soup.find('title').text)
        print(url)
    except:
        print(f'Page has no title')
        print(url)

# TODO
def get_last_modified(response):
    # get the head of the response and check if it has Last-Modified tag
    last_modified = response.headers.get('Last-Modified')

    if last_modified:
        print(f'Last-Modified: {last_modified}')
    else:
        print('No Last-Modified information in page head')
    
    print()

    return True


# crawls the given url recursively
def crawl(url):
    if url in visited:
        print('URL has already been visited')
        return

    visited.add(url)

    # check robots.tsx
    if not parsing_allowed(url):
        print('URL not allowed')

        return

    response = requests.get(url, headers={'User-Agent': CRAWLER_NAME})

    # check if website is available
    if response.status_code != 200:
        print('URL returns wrong code')
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # do something with the content
    process_page(url, soup)
    get_last_modified(response)

    for link in soup.find_all('a', href=True):

        # just returns link['href'] is the link is absolute
        next_url = urljoin(url, link['href'])

        if next_url.startswith('http'):
            time.sleep(get_crawl_delay(url))
            if next_url not in visited:
                crawl(next_url)


def main():
    crawl(seeds[0])
    print(f'Number of visited sites: {len(visited)}')


if __name__ == '__main__':
    main()
