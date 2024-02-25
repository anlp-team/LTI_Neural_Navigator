from unstructured.partition.html import partition_html
from unstructured.partition.pdf import partition_pdf


class preprocessor:
    def __init__(self, file_path: str = None, url: str = None):
        self.file_path = file_path
        self.url = url

        self.tags_ignore = [
            # "script",
            # "style",
            # "meta",
            # "link",
            # "noscript",
            # "img",
            # "svg",
            # "path",
            "a",
            "header",
            "footer",
            "nav",
            "input",
            "form",
            "button",
            "address",
        ]

    def parse_html(self, url: str = None, file_path: str = None):
        if file_path is not None:
            elements = partition_html(filename=file_path)
        elif url is not None:
            elements = partition_html(url=url)
        else:
            raise ValueError("Either file_path or url must be provided")

        self.html_elements = elements

        return elements

    def parse_pdf(self, file_path: str = None):
        elements = partition_pdf(filename=file_path)
        for element in elements:
            print(element.__dict__)
            print(element.metadata.__dict__)
            print(element, "\n\n")
        raise NotImplementedError

        self.pdf_elements = elements
        return elements

    def process_elements(self, elements: list = None):
        texts = []
        for element in elements:
            tags = list(element.ancestortags) + [element.tag]
            text = " ".join(element.text.split())
            if any(tag in self.tags_ignore for tag in tags):
                print(f"Skipping: {text[:20]} ... {text[-20:]}")
                continue
            texts.append(text)

        return "\n\n".join(texts)

    def process_pdf(self, file_path: str = None):
        pass

    def print_elements(self):
        self.html_elements = partition_html(url=self.url)
        for element in self.html_elements:
            print(element.__dict__)
            print(element.metadata.__dict__)
            print(element, "\n\n")


if __name__ == "__main__":
    p = preprocessor(url="https://www.cs.cmu.edu/scs25/25things")
    p.print_elements()