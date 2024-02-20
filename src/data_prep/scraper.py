from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import utils


class Scraper:

    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(options=self.chrome_options)
        # self.driver.implicitly_wait(10)
        self.driver.set_page_load_timeout(10)

    def parse(self, url: str):
        soup = self.get_soup(url)
        soup = self.remove_js_css(soup)
        # TODO: should I remove more stuff?
        return soup

    def get_html(self, url: str):
        self.driver.get(url)
        return self.driver.page_source

    def get_soup(self, url: str):
        html = self.get_html(url)
        return BeautifulSoup(html, "html.parser")

    def get_links(self, soup: BeautifulSoup):
        # only anchor tag links to external pages. Instead link tag link doc to the current page
        links = [link.get("href") for link in soup.find_all("a")]
        return [link for link in links if link is not None and link.startswith("http")]

    def get_title(self, soup: BeautifulSoup):
        if soup.title is None:
            return f"untitsled_{utils.get_timestamp()}"
        title = soup.title.string.replace(" ", "_").replace("/", "__")
        return title

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

    def get_date(self, date_str: str):
        return datetime.strptime(date_str, "%Y-%m-%d")

    def close(self):
        self.driver.quit()

    def __del__(self):
        self.close()


if __name__ == "__main__":
    scraper_ = Scraper()
    soup = scraper_.get_soup("https://www.cs.cmu.edu/scs25/25things")
    # print(soup)
    # print(scraper_.remove_js_css(soup))
    # print(scraper_.get_links(soup))
