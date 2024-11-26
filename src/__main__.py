# https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/#running-a-command-line-interface-from-source-with-src-layout
import os
import sys

if not __package__:
    # Make CLI runnable from source tree with
    #    python src/package
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    sys.path.insert(0, package_source_path)


from src import paper_crawler  # noqa: F401, RUF100
from src import paper_summary  # noqa: F401, RUF100


def main():
    print("Hello from simple-paper-summarization!")
    print(paper_crawler.hello())
    print(paper_summary.hello())


if __name__ == "__main__":
    main()
