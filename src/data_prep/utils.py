import os
import json
from datetime import datetime
from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Union, List
from tqdm import tqdm
from selenium.common.exceptions import TimeoutException, WebDriverException


def save_html(soup: BeautifulSoup, path: str, page_title: str = None):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/{page_title}.html"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.exists(path):
        path = f"{path[:-5]}_{get_timestamp()}.html"  # some pages have the same title

    with open(path, "w") as file:
        file.write(str(soup))


def bfs_pages(
    scraper,
    url: Union[str, List[str]],
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
        if u in visited:
            continue
        tqdm.write(f"Visiting {u}")

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
        visited[u] = scraper.get_title(soup)
        save_html(soup, "data/raw", visited[u])

    bfs_pages(scraper, links[29:], visited, depth + 1)


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
