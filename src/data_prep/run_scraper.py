import scraper
from collections import defaultdict
from utils import bfs_pages, save_visited_json, load_visited_json


def main():
    data_path = "./data/raw"
    visited = load_visited_json(data_path)

    scraper_ = scraper.Scraper()
    url = "https://www.cs.cmu.edu/scs25/25things"
    bfs_pages(scraper_, url, visited)
    scraper_.close()

    save_visited_json(visited, data_path)


if __name__ == "__main__":
    main()
