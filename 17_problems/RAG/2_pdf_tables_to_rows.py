# pip install pdfplumber langchain-community langchain-text-splitters rank_bm25
import os
import re
import sys

import pdfplumber
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# PROBLEM: a PDF loader turns a table into a stream of words. Table 1.1 in
# the book (Semester | Instructor | Class Size | Rating) comes out as:
#       Semester Instructor Class Size Rating
#       Fall 2020 A 100
#       Not
#       recommended
#       at all
#       Spring 2021 A 50 Highly
#       ...
#       Table 1.1 Course Evaluation Records      <- the caption comes AFTER the rows
# Why it hurts RAG:
#   1. Cells that wrap onto several lines get mixed up with the next cells,
#      so "which value belongs to which column?" is lost.
#   2. Once chunked, a chunk can hold rows like "Spring 2022 B 40 Highly"
#      with no column names and no caption: the LLM has to guess what 40 is.
#   3. A question like "rating of the class of 40 students" must match the
#      words "Rating" and "Class Size", which only appear once, in the header.
#
# SOLUTION (pdfplumber + LangChain Documents):
#   1. pdfplumber finds tables from the ruling lines on the page and returns
#      rows and cells. Only "real" tables are kept (at least 2 columns and
#      3 rows) - this book also draws boxes around notes and code, which
#      pdfplumber reports as 1-column tables.
#   2. Each ROW becomes its own Document, written so it stands on its own:
#         Table 1.1 Course Evaluation Records | Semester: Spring 2022 |
#         Instructor: B | Class Size: 40 | Rating: Highly recommended
#      The column names and the caption are repeated in every row, so the row
#      is understandable (and findable) no matter how it is chunked.
#   3. The page's normal text is taken from OUTSIDE the table area, so the
#      jumbled version of the table is not indexed a second time.
# A BM25 keyword retriever (no API key) compares "plain" vs "table-aware".
#
# Run it:
#   python 2_pdf_tables_to_rows.py                    pages 20-110
#   python 2_pdf_tables_to_rows.py --all              the whole book (about 2 minutes)
#   python 2_pdf_tables_to_rows.py "your question"
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "..", "9_general", "rag", "Principles-of-Data-Science.pdf")
PAGES = range(1, 570) if "--all" in sys.argv else range(20, 111)   # 1-based PDF page numbers

MIN_ROWS, MIN_COLS = 3, 2          # smaller "tables" are boxed notes or code, not data tables
CAPTION = re.compile(r"^Table\s+[\w.]+\s+.+")   # "Table 1.1 Course Evaluation Records"


def cell(text) -> str:
    return " ".join((text or "").split())     # merged cells come back as None; wrapped cells contain \n


def caption_below(page, bbox) -> str:
    """OpenStax puts the caption just under the table."""
    top, bottom = bbox[3], min(bbox[3] + 40, page.height)
    text = page.crop((0, top, page.width, bottom)).extract_text() or ""
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if CAPTION.match(first) else f"Table on page {page.page_number}"


def row_documents(page, table) -> list[Document]:
    rows = [[cell(c) for c in row] for row in table.extract()]
    header, body = rows[0], rows[1:]
    caption = caption_below(page, table.bbox)
    docs = []
    for row in body:
        if not any(row):
            continue
        pairs = [f"{h or f'Column {i + 1}'}: {v}" for i, (h, v) in enumerate(zip(header, row)) if v]
        docs.append(Document(page_content=f"{caption} | " + " | ".join(pairs),
                             metadata={"page": page.page_number, "type": "table_row", "table": caption}))
    return docs


def load(pdf_path: str) -> tuple[list[Document], list[Document], int]:
    """Returns (plain page documents, table-aware documents, number of tables found)."""
    plain, aware, n_tables = [], [], 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_no in PAGES:
            page = pdf.pages[page_no - 1]
            plain.append(Document(page_content=page.extract_text() or "", metadata={"page": page_no}))

            tables = [t for t in page.find_tables()
                      if len(t.rows) >= MIN_ROWS and len(t.rows[0].cells) >= MIN_COLS]
            prose_page = page
            for table in tables:
                aware += row_documents(page, table)
                prose_page = prose_page.outside_bbox(table.bbox)    # cut the table out of the page text
            aware.append(Document(page_content=prose_page.extract_text() or "",
                                  metadata={"page": page_no, "type": "text"}))
            n_tables += len(tables)
    return plain, aware, n_tables


def retriever(docs: list[Document]) -> BM25Retriever:
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(docs)
    # BM25's default tokenizer is str.split(): "Rating" and "rating" would not match. Lower-case words instead.
    return BM25Retriever.from_documents(chunks, k=2, preprocess_func=lambda t: re.findall(r"\w+", t.lower()))


def main() -> None:
    print(f"Reading pages {PAGES.start}-{PAGES.stop - 1} of {os.path.basename(PDF_PATH)} with pdfplumber ...")
    plain, aware, n_tables = load(PDF_PATH)
    n_rows = sum(d.metadata.get("type") == "table_row" for d in aware)
    print(f"\n1. Found {n_tables} data tables -> {n_rows} row documents")

    print("\n2. TABLE 1.1, AS A PLAIN LOADER SEES IT (page 27):")
    text = next(d.page_content for d in plain if d.metadata["page"] == 27)
    start = text.find("Semester Instructor")
    print("   " + text[start:text.find("Table 1.1 Course", start) + 40].replace("\n", "\n   "))

    print("\n   ...AND AS ROW DOCUMENTS:")
    for d in [d for d in aware if d.metadata.get("table", "").startswith("Table 1.1")][:4]:
        print("   " + d.page_content)

    plain_r, aware_r = retriever(plain), retriever(aware)
    args = [a for a in sys.argv[1:] if a != "--all"]
    questions = [" ".join(args)] if args else ["What was the rating of the class with a class size of 40 in Spring 2022?",
                                               "benefit of data security in a DBMS"]
    for q in questions:
        print(f"\n3. RETRIEVAL (BM25, top 2) for: \"{q}\"")
        for name, r in (("plain", plain_r), ("table-aware", aware_r)):
            print(f"  {name}:")
            for doc in r.invoke(q):
                print(f"    p.{doc.metadata['page']:<4} {' '.join(doc.page_content.split())[:150]}...")


if __name__ == "__main__":
    main()
