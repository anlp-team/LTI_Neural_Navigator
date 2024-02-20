import os
import json
import urllib.request

from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Union, List
from collections import defaultdict, Iterator
from selenium.common.exceptions import TimeoutException, WebDriverException


def save_html(
    soup: BeautifulSoup, path: str, page_title: str = None, delimeter: str = " "
):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/{page_title}.html"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.exists(path):
        path = f"{path[:-5]}_{get_timestamp()}.html"  # some pages have the same title

    text = delimeter.join(soup.stripped_strings)
    with open(path, "w") as file:
        file.write(text)


def save_pdf(url: str, path: str):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/{url.split('/')[-1].replace('%', '_')}"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        urllib.request.urlretrieve(
            url, path
        )  # TODO: assuming no pdfs have the same name
    except Exception as e:  # TODO: HTTP Error 403: forbidden
        tqdm.write(f"Error: {e}")
        return
    return url.split("/")[-1].replace("%", "_")


def bfs_pages(
    scraper,
    url: Union[
        str, List[str], Iterator[str]
    ],  # Iterator[str] or Generator[str, None, None] (Generator[YieldType, SendType, ReturnType])
    visited: defaultdict[str, str],
    depth: int = 0,
    max_depth: int = 2,
):
    if depth >= max_depth:
        return
    tqdm.write(f"Depth: {depth + 1}")

    links = []
    if isinstance(url, str):
        url = [url]

    for u in tqdm(url):
        # if u in visited:  # TODO: should skip here or after parsing?
        #     continue

        tqdm.write(f"Visiting {u}")

        # if the page is a pdf download it to the data/raw folder
        if u.endswith(".pdf"):
            try:
                visited[u] = save_pdf(
                    u, "data/raw"
                )  # TODO: will pdf contain any useful links?
            except Exception as e:
                tqdm.write(f"Error: {e}")
            continue

        # otherwise parse the page and get the links
        try:
            soup = scraper.parse(u)
        except TimeoutException as e:
            tqdm.write(f"TimeoutException: {e}")
            continue
        except WebDriverException as e:
            tqdm.write(f"WebDriverException: {e}")
            continue

        links.extend(
            scraper.get_links(soup)
        )  # TODO: should I filter out only cmu links?

        if u not in visited:  # only save the page if it hasn't been visited before
            visited[u] = scraper.get_title(soup)
            save_html(soup, "data/raw", visited[u])

    bfs_pages(scraper, links, visited, depth + 1, max_depth)


def save_visited_json(visited: defaultdict[str, str], path: str):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/visited.json"
    if not os.path.exists(os.path.dirname(path)):
        raise FileNotFoundError(f"Path {path} does not exist")

    with open(path, "w") as file:
        json.dump(visited, file)


def load_visited_json(path: str):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/visited.json"
    if not os.path.exists(path):
        return defaultdict(str)

    with open(path, "r") as file:
        return json.load(file)


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
