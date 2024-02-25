from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import pypdf


class scraper:

    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=self.chrome_options)
        # self.driver.implicitly_wait(10)
        self.driver.set_page_load_timeout(
            5
        )  # TODO: is this too short? what causes a page to load for more than 10 seconds?

        self.current_url = None

    def set_domain(self, url: str):
        self.current_url = url.split("/")[2]

    def fetch(self, url: str, raw_html: bool = False):
        soup = self.get_soup(url)
        if not raw_html:
            soup = self.remove_js_css(soup)
        # TODO: should I remove more stuff?
        return soup, self.get_links(soup), self.get_title(soup)

    def get_html(self, url: str):
        # set the current url to the domain of the page
        self.set_domain(url)
        self.driver.get(url)
        return self.driver.page_source

    def get_soup(self, url: str):
        html = self.get_html(url)
        return BeautifulSoup(html, "html.parser")

    def get_links(self, soup: BeautifulSoup):
        # only anchor tag links to external pages. Instead link tag link doc to the current page
        links = [link.get("href") for link in soup.find_all("a")]
        return self.filter_links(links)

    def get_title(self, soup: BeautifulSoup):
        from utils import get_timestamp

        if soup.title is None:
            return f"untitsled_{get_timestamp()}"
        title = soup.title.string.replace(" ", "_").replace("/", "__")
        return title.replace("\n", "")

    def remove_js_css(self, soup: BeautifulSoup):
        # TODO: is this too aggressive?
        for script in soup(["script", "style"]):
            script.extract()
        return soup

    def remove_attrs(self, soup: BeautifulSoup, attrs: list):
        for tag in soup.find_all(True):
            for attr in attrs:
                del tag[attr]
        return soup

    def remove_tags(self, soup: BeautifulSoup, tags: list):
        for tag in tags:
            for match in soup.find_all(tag):
                match.decompose()
        return soup

    def filter_links(self, links: list):
        for link in links:
            if link is None:
                continue
            if link.startswith("http"):
                yield link
            if link.startswith("/"):
                yield f"https://{self.current_url}{link}"  # some links are relative to the current page

    def get_date(self, date_str: str):
        return datetime.strptime(date_str, "%Y-%m-%d")

    def close(self):
        self.driver.quit()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    scraper_ = scraper()
    soup = scraper_.get_soup("https://www.cs.cmu.edu/scs25/25things")
    # print(soup)
    # print(scraper_.remove_js_css(soup))
    print(list(scraper_.get_links(soup)))
