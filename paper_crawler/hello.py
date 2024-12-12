import urllib.parse
from playwright.sync_api import sync_playwright, expect
import urllib
from urllib.parse import urlparse
import enum
import pathlib


class ARXIV_SEARCH_TYPE(enum.Enum):
    ALL = "all"


CACHE_DIR = pathlib.Path(__file__).parent.joinpath(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"  # noqa: E501
        )
        original_parameters = {
            "query": "llm",
            "searchtype": ARXIV_SEARCH_TYPE.ALL.value,
            "abstracts": "show",
            "order": "-announced_date_first",
            "size": 25,  # 25、50、100、200
        }
        parameters = urllib.parse.urlencode(original_parameters)
        print(parameters)
        page.goto(f"https://arxiv.org/search/?{parameters}")
        content_locator = page.locator("#main-container > div.content > ol")
        expect(content_locator).to_be_visible(visible=True)
        paper_url_set: list[str] = []
        with CACHE_DIR.joinpath("list.txt").open(mode="w", encoding="utf-8") as f:
            f.write("")
            items = content_locator.locator("li")
            for i in range(items.count()):
                locator = items.nth(i).locator("div > p > a")
                expect(locator).to_be_visible(visible=True)
                href = locator.get_attribute("href")
                if href is not None:
                    paper_url_set.append(href)
                    f.write(href)
                    f.write("\n")
                # f.write("\n--------------------------------------------------")
        for paper_url in paper_url_set:
            print(paper_url)
            url = urlparse(paper_url)
            page.goto(paper_url)
            title_locator = page.locator("#abs > h1")
            expect(title_locator).to_be_visible(visible=True)
            cache_key_path = CACHE_DIR.joinpath(str(url.path[1:]))
            print(cache_key_path.absolute())
            cache_key_path.mkdir(exist_ok=True, parents=True)
            # title -----------------------------------------------
            cache_key_path.joinpath("title.txt").write_text(
                encoding="utf-8",
                data=title_locator.text_content()
                .strip()
                .removeprefix("Title:")
                .strip(),
            )
            # author -----------------------------------------------
            author_set_locator = page.locator("#abs > div.authors")
            expect(author_set_locator).to_be_visible(visible=True)
            with cache_key_path.joinpath("authors.txt").open(
                mode="w", encoding="utf-8"
            ) as f:
                author_set_locator = author_set_locator.locator("a")
                for i in range(author_set_locator.count()):
                    author_locator = author_set_locator.nth(i)
                    expect(author_locator).to_be_visible(visible=True)
                    f.write(f"{author_locator.text_content().strip()}\n")
            # abstract -----------------------------------------------
            abstract_set_locator = page.locator("#abs > blockquote")
            expect(abstract_set_locator).to_be_visible(visible=True)
            cache_key_path.joinpath("abstract.txt").write_text(
                encoding="utf-8",
                data=abstract_set_locator.text_content()
                .strip()
                .removeprefix("Abstract:")
                .strip(),
            )
            print("------------------------------------------------------------")

        # input()
        browser.close()


if __name__ == "__main__":
    main()
