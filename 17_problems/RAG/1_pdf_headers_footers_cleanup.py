# pip install langchain-community langchain-text-splitters pypdf rank_bm25
import os
import re
import sys
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# PROBLEM: a real PDF is not just content. A loader such as PyPDFLoader
# gives you EVERYTHING on every page:
#   - title, copyright and "buy our app" pages at the front
#   - a table of contents and an index (pages full of "topic ... 187")
#   - a running header/footer on every page, e.g.
#         "12     1 • What Are Data and Data Science?"
#         "Access for free at openstax.org"
#
# Why it hurts RAG:
#   1. A footer ends up in the MIDDLE of a chunk when a sentence continues
#      on the next page, and it is embedded with the content.
#   2. Footers repeat on hundreds of pages, so the words in them
#      ("Linear Regression Analysis") match queries on pages that are not
#      really about that topic.
#   3. TOC and index pages mention every topic and get retrieved for almost
#      any question - with no actual explanation in them.
#
# SOLUTION (no hard-coded strings for this book - it works on other PDFs):
#   1. Header/footer lines are found by REPETITION. Look at the top 2 and
#      bottom 3 lines of every page, replace digits with 0 (so "Preface 3"
#      and "Preface 5" look the same) and count. A line shape is boilerplate
#      if it repeats on several pages AND either carries a page number at its
#      start/end, or appears on a big share of all pages.
#      (Repetition alone is not enough: content labels such as "PYTHON CODE"
#      or "EXAMPLE 3.2" also repeat at page edges.)
#   2. Front matter = everything before the first page with a page-numbered
#      header/footer (cover, title, copyright, ads, TOC have none).
#   3. List pages (TOC, index): most lines end in a page number.
#   4. Remove the boilerplate lines (only at page edges), then drop pages
#      that are (almost) empty afterwards.
# Then chunk as usual. A BM25 keyword retriever (no API key, no embeddings)
# shows the effect on retrieval: noisy vs clean.
#
# Run it:
#   python 1_pdf_headers_footers_cleanup.py
#   python 1_pdf_headers_footers_cleanup.py "your search query"
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "..", "9_general", "rag", "Principles-of-Data-Science.pdf")

EDGE_TOP, EDGE_BOTTOM = 2, 3   # lines at the top / bottom of a page where headers/footers live
MIN_REPEAT = 3                 # a header/footer shape must appear on at least this many pages
FREQUENT_SHARE = 0.25          # ...or on this share of all pages (catches lines without a page number)
LIST_PAGE_SHARE = 0.5          # TOC/index: more than half the lines end in a page number
MIN_CONTENT_CHARS = 100        # a page with less text than this after cleaning is dropped

PAGE_NUMBER = re.compile(r"^0( |$)|( |^)0$")         # a shape that starts or ends with a page number
LIST_LINE = re.compile(r"\s\d{1,3}(\s*[A-Z])?\s*$")   # "4.2 Hypothesis Testing 165", "hashing  391"


def lines_of(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


def shape(line: str) -> str:
    """'12     1 • What Are Data?' -> '0 0 • What Are Data?' so every page's footer looks the same."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", line)).strip()


def is_edge(index: int, n_lines: int) -> bool:
    return index < EDGE_TOP or index >= n_lines - EDGE_BOTTOM


def find_boilerplate(pages: list[Document]) -> set[str]:
    counts = Counter()
    for page in pages:
        lines = lines_of(page.page_content)
        counts.update({shape(line) for i, line in enumerate(lines) if is_edge(i, len(lines))})
    return {s for s, c in counts.items()
            if c >= MIN_REPEAT and re.search(r"[A-Za-z]", s)
            and (PAGE_NUMBER.search(s) or c >= FREQUENT_SHARE * len(pages))}


def is_list_page(lines: list[str]) -> bool:
    return len(lines) >= 10 and sum(bool(LIST_LINE.search(l)) for l in lines) / len(lines) > LIST_PAGE_SHARE


def has_page_number_footer(lines: list[str], boilerplate: set[str]) -> bool:
    return any(is_edge(i, len(lines)) and shape(l) in boilerplate and PAGE_NUMBER.search(shape(l))
               for i, l in enumerate(lines))


def clean_pages(pages: list[Document]) -> tuple[list[Document], dict[int, str], set[str]]:
    """Returns the cleaned pages, {page index: why it was dropped}, and the header/footer shapes."""
    boilerplate = find_boilerplate(pages)
    first_body = next(i for i, p in enumerate(pages) if has_page_number_footer(lines_of(p.page_content), boilerplate))

    kept, dropped = [], {}
    for i, page in enumerate(pages):
        lines = lines_of(page.page_content)
        if i < first_body:
            dropped[i] = "front matter (before page numbering starts)"
            continue
        if is_list_page(lines):
            dropped[i] = "table of contents / index"
            continue
        body = [l for j, l in enumerate(lines) if not (is_edge(j, len(lines)) and shape(l) in boilerplate)]
        text = "\n".join(body)
        if len(text) < MIN_CONTENT_CHARS:
            dropped[i] = "(almost) empty once header/footer removed"
            continue
        kept.append(Document(page_content=text, metadata=page.metadata))
    return kept, dropped, boilerplate


def chunk(pages: list[Document]) -> list[Document]:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)


def count_noisy(chunks: list[Document], boilerplate: set[str]) -> int:
    return sum(any(shape(l) in boilerplate for l in lines_of(c.page_content)) for c in chunks)


def show_hits(title: str, retriever: BM25Retriever, query: str, dropped: dict[int, str]) -> None:
    print(f"  {title}")
    for doc in retriever.invoke(query):
        snippet = " ".join(doc.page_content.split())[:80]
        junk = f"   <- JUNK: {dropped[doc.metadata['page']]}" if doc.metadata["page"] in dropped else ""
        print(f"    p.{doc.metadata['page'] + 1:<4} {snippet}...{junk}")


def main() -> None:
    print(f"Loading {os.path.basename(PDF_PATH)} ...")
    pages = PyPDFLoader(PDF_PATH).load()        # one Document per PDF page
    clean, dropped, boilerplate = clean_pages(pages)

    print(f"\n1. PAGES: {len(pages)} loaded, {len(clean)} kept")
    for reason, n in Counter(dropped.values()).items():
        print(f"   dropped {n:>3}  {reason}")

    print(f"\n2. HEADER/FOOTER SHAPES FOUND: {len(boilerplate)} (digits shown as 0). The most frequent:")
    counts = Counter(shape(l) for p in pages for l in lines_of(p.page_content) if shape(l) in boilerplate)
    for s, n in counts.most_common(6):
        print(f"   {n:>4} pages  {s}")

    print("\n3. ONE PAGE, BEFORE and AFTER (last 3 lines of PDF page 22):")
    before = lines_of(pages[21].page_content)[-3:]
    after = lines_of(next(p for p in clean if p.metadata["page"] == 21).page_content)[-3:]
    print("   before: " + "\n           ".join(before))
    print("   after:  " + "\n           ".join(after))

    noisy_chunks, clean_chunks = chunk(pages), chunk(clean)
    print(f"\n4. CHUNKS: noisy {len(noisy_chunks)}, clean {len(clean_chunks)}")
    print(f"   chunks containing a header/footer line: noisy {count_noisy(noisy_chunks, boilerplate)}, "
          f"clean {count_noisy(clean_chunks, boilerplate)}")

    # BM25 = keyword search: no API key, no embeddings. The same junk pages hurt vector search too.
    noisy_retriever = BM25Retriever.from_documents(noisy_chunks, k=3)
    clean_retriever = BM25Retriever.from_documents(clean_chunks, k=3)
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else ["hallucinations", "R Studio", "hashing"]
    for query in queries:
        print(f"\n5. RETRIEVAL (BM25, top 3) for: \"{query}\"")
        show_hits("noisy:", noisy_retriever, query, dropped)
        show_hits("clean:", clean_retriever, query, dropped)


if __name__ == "__main__":
    main()
