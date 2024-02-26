import os
import json
import pypdf
import urllib.request

from tqdm import tqdm
from datetime import datetime
from bs4 import BeautifulSoup
from typing import Union, List

# from scraper import scraper
# from preproessor import preprocessor
from collections import defaultdict
from collections.abc import Iterator
from selenium.common.exceptions import TimeoutException, WebDriverException


def bfs_pages(
    scraper_: "scraper",
    urls: Union[
        str, List[str], Iterator[str]
    ],  # Iterator[str] or Generator[str, None, None] (Generator[YieldType, SendType, ReturnType])
    visited: defaultdict[str, str],
    depth: int = 0,
    max_depth: int = 2,
    all_html_paths: List[str] = [],
    all_pdf_paths: List[str] = [],
    raw_html: bool = False,
):
    if depth >= max_depth:
        return []
    tqdm.write(f"Depth: {depth + 1}")

    links = []
    if isinstance(urls, str):
        urls = [urls]

    for u in tqdm(urls):
        # if (
        #     u in visited
        # ):  # TODO: should skip here or after parsing? Here if I assume the page is visited during this execution
        #     continue

        tqdm.write(f"Visiting {u}")

        # if the page is a pdf download it to the data/raw/bs folder
        if u.endswith(".pdf"):
            visited[u] = u.split("/")[-1].replace("%", "_").replace(".pdf", "")
            all_pdf_paths.append(
                save_pdf(u, "data/raw/bs")
            )  # TODO: will pdf contain any useful links?
            continue

        if u.endswith(".xlsx") or u.endswith(".ics"):
            visited[u] = u.split("/")[-1].replace("%", "_")
            save_non_html(u, "data/raw/bs")
            continue

        # otherwise parse the page and get the links
        try:
            soup_u, links_u, title_u = scraper_.fetch(
                u, raw_html=raw_html
            )  # TODO: .html***.html bug
        except TimeoutException as e:
            tqdm.write(f"TimeoutException: {e}")
            continue
        except WebDriverException as e:
            tqdm.write(f"WebDriverException: {e}")
            continue

        links.extend(list(links_u))  # TODO: should I filter out only cmu links?

        if u not in visited:  # only save the page if it hasn't been visited before
            visited[u] = title_u
            extension = "html" if raw_html else "txt"
            all_html_paths.append(
                save_html(soup_u, "data/raw/bs", visited[u], extention=extension)
            )

    links.extend(
        bfs_pages(
            scraper_,
            links,
            visited,
            depth + 1,
            max_depth,
            all_html_paths,
            all_pdf_paths,
            raw_html,
        )
    )
    return links


def preprocess_unstructured(
    preprocessor_: "preprocessor", all_html_paths: List[str], all_pdf_paths: List[str]
):
    for path in tqdm(all_html_paths):
        print(f"Processing {path}")
        try:
            elements = preprocessor_.parse_html(file_path=path)
            text = preprocessor_.process_elements(elements)
        except Exception as e:
            print(f"Error: {e} | {path}")
        save_str(
            text, "data/raw/unstruct", path.split("/")[-1].replace(".html", ""), "html"
        )

    for path in tqdm(all_pdf_paths):
        print(f"Processing {path}")
        if path is None:
            continue
        try:
            elements = preprocessor_.parse_pdf(
                file_path=path, include_page_breaks=False
            )
            text = preprocessor_.process_pdf(elements)
        except Exception as e:
            print(f"Error: {e} | {path}")
        save_str(
            text, "data/raw/unstruct", path.split("/")[-1].replace(".pdf", ""), "pdf"
        )


def save_html(
    soup: BeautifulSoup,
    path: str,
    page_title: str = None,
    delimeter: str = " ",
    extention: str = "txt",
):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/html/{page_title}.{extention}"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.exists(path):
        path = f"{path[:-4]}_{get_timestamp()}.{extention}"  # some pages have the same title

    if extention == "txt":
        text = delimeter.join(soup.stripped_strings)
    elif extention == "html":
        text = soup.prettify()
    with open(path, "w") as file:
        file.write(text)

    return path


def save_pdf(url: str, path: str):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/pdf/{url.split('/')[-1].replace('%', '_').replace('.pdf', '')}"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        urllib.request.urlretrieve(
            url, f"{path}.pdf"
        )  # TODO: assuming no pdfs have the same name
    except Exception as e:  # TODO: HTTP Error 403: forbidden
        tqdm.write(f"Error: {e}")
        return

    # extract the text from the pdf
    reader = pypdf.PdfReader(f"{path}.pdf")
    with open(f"{path}.txt", "w") as file:
        for page in reader.pages:
            text = page.extract_text()
            file.write(text)

    # remove the pdf
    # os.remove(f"{path}.pdf")

    return f"{path}.pdf"


def save_non_html(url: str, path: str):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/non_html/{url.split('/')[-1].replace('%', '_')}"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    try:
        urllib.request.urlretrieve(url, f"{path}.{url.split('.')[-1]}")
    except Exception as e:
        tqdm.write(f"Error: {e}")
        return

    return f"{path}.{url.split('.')[-1]}"


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


def save_str(
    html: str, path: str, page_title: str = None, original_extention: str = "html"
):
    datetime_str = datetime.now().strftime("%Y-%m-%d")
    path = f"{path}/{datetime_str}/{original_extention}/{page_title}.txt"

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.exists(path):
        path = f"{path[:-4]}_{get_timestamp()}.txt"  # some pages have the same title

    with open(path, "w") as file:
        file.write(html)


def filter_txt_files(path: str, extention: str = "txt"):
    # iterate through all the files in the directory
    # and filter out the ones that do not contain the given keywords
    keywords = [
        "cmu",
        "carnegie",
        "mellon",
        "university",
        "tartans",
        "scotty",
        "pittsburgh",
        "carnival",
        "CMU",
        "Carnegie",
        "Mellon",
        "University",
        "Tartans",
        "Scotty",
        "Pittsburgh",
        "Carnival",
    ]

    files = []
    for root, _, file_names in os.walk(path):
        for file_name in file_names:
            if file_name.endswith(extention):
                files.append(os.path.join(root, file_name))

    filtered_files = []
    for file in files:
        with open(file, "r") as f:
            text = f.read()
            if any(keyword in text for keyword in keywords) or any(
                keyword in file for keyword in keywords
            ):
                filtered_files.append(file)
            else:
                print(f"Removing {file} | no keywords found in file")
                os.remove(file)
                continue

            # if the file is less than 100 characters, remove it
            if len(text) < 100:
                print(f"Removing {file} | {len(text)} characters")
                os.remove(file)

            if len(text) > 100 and len(text) < 200:
                print(f" {file} | {len(text)} characters")
                os.remove(file)

            if "Page_not_found" in file:
                print(
                    f"Removing {file} | Page_not_found -> len(text) {len(text)} characters"
                )
                os.remove(file)

    return filtered_files


if __name__ == "__main__":
    filter_txt_files("data/raw/unstruct/2024-02-26/html", "txt")
