import scraper
from collections import defaultdict
from utils import bfs_pages, save_visited_json, load_visited_json
import time


list_of_sources = [
    "https://lti.cs.cmu.edu/directory/all/154/1",
    "https://enr-apps.as.cmu.edu/open/SOC/SOCServlet/completeSchedule",
    "https://www.cmu.edu/hub/calendar/",
    "https://lti.cs.cmu.edu/learn",
    "https://web.cvent.com/event/ab7f7aba-4e7c-4637-a1fc-dd1f608702c4/websitePage:645d57e4-75eb-4769-b2c0-f201a0bfc6ce?locale=en",
    "https://www.cmu.edu/commencement/schedule/index.html",
    "https://www.cs.cmu.edu/scs25/25things",
    "https://www.cs.cmu.edu/scs25/history",
    "https://www.cmu.edu/about/history.html",
    "https://www.cmu.edu/news/stories/archives/2019/april/spring-carnival-buggy.html",
    "https://athletics.cmu.edu/athletics/tartanfacts",
    "https://athletics.cmu.edu/athletics/mascot/about",
    "https://athletics.cmu.edu/athletics/kiltieband/index",
]


def main():
    data_path = "./data/raw"
    visited = load_visited_json(data_path)

    start = time.time()

    scraper_ = scraper.Scraper()
    for url in list_of_sources:
        bfs_pages(scraper_, url, visited)
    scraper_.close()

    print(f"Time: {time.time() - start:.2f}")

    save_visited_json(visited, data_path)


if __name__ == "__main__":
    main()
