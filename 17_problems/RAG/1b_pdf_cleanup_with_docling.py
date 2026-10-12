# Docling brings PyTorch and layout models - install it in a SEPARATE venv, not the repo's shared .venv:
#   uv venv docling-venv
#   uv pip install --python docling-venv "docling==2.135.0" "langchain-docling==1.1.0" "langchain-core==0.3.86" \
#       "langchain-community==0.3.31" "langchain-text-splitters==0.3.11" "pypdf==6.14.2" "numpy==2.4.6"
# (langchain-docling 3.x needs langchain-core 1.x; 1.1.0 is the version that works with this repo's pins.)
import importlib.util
import os
import re
import sys
import time

os.environ["TRANSFORMERS_VERBOSITY"] = "error"   # hides a harmless "sequence length > 512" tokenizer warning

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.document_loaders import PyPDFLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# PROBLEM 1b: same problem as 1_pdf_headers_footers_cleanup.py, solved with
# an open-source library instead of hand-written rules.
#
# Docling (IBM, MIT licence) runs a LAYOUT MODEL on every page and labels
# each region: title, text, list item, table, page_header, page_footer,
# document_index (the book's index)... Headers and footers go to a separate
# "furniture" layer, so DoclingLoader never puts them into chunks.
#
# What Docling does NOT do: drop the cover/title/copyright pages or the table
# of contents - to the model those are ordinary text and lists. So we still
# add two small filters: chunks labelled document_index, and list-like chunks
# (most lines end in a page number - the same check as script 1).
#
# Bonus: Docling chunks follow the document's structure, not PDF pages, so a
# sentence that continues on the next page stays in one chunk.
#
# This script runs BOTH methods on the same pages and compares them.
# By default it uses pages 1-30 (cover, front matter, TOC, preface, start of
# chapter 1) and 560-569 (index) - about 2-3 minutes on a CPU.
#
# Run it (from the docling venv):
#   python 1b_pdf_cleanup_with_docling.py          sample pages
#   python 1b_pdf_cleanup_with_docling.py --all    the whole book (~15-20 minutes on a CPU)
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "..", "..", "9_general", "rag", "Principles-of-Data-Science.pdf")
PAGE_RANGES = [(1, 569)] if "--all" in sys.argv else [(1, 30), (560, 569)]   # 1-based, inclusive

LIST_LINE = re.compile(r"\s\d{1,3}(\s*[A-Z])?\s*$")   # same TOC/index check as script 1


def load_rules_module():
    """Script 1 starts with a digit, so it cannot be imported normally."""
    spec = importlib.util.spec_from_file_location("rules", os.path.join(BASE_DIR, "1_pdf_headers_footers_cleanup.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def in_ranges(page_no: int) -> bool:          # 1-based page number
    return any(start <= page_no <= end for start, end in PAGE_RANGES)


def docling_chunks() -> list:
    # No OCR (the PDF has a text layer) and no table-structure model: much faster on a CPU.
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    chunks = []
    for page_range in PAGE_RANGES:
        loader = DoclingLoader(PDF_PATH, converter=converter, convert_kwargs={"page_range": page_range},
                               export_type=ExportType.DOC_CHUNKS)
        chunks += loader.load()
    return chunks


def chunk_info(chunk) -> tuple[set[str], set[int]]:
    """The layout labels and the (1-based) page numbers a Docling chunk came from."""
    items = chunk.metadata["dl_meta"]["doc_items"]
    return {i["label"] for i in items}, {p["page_no"] for i in items for p in i["prov"]}


def is_toc_or_index(chunk) -> bool:
    labels, _ = chunk_info(chunk)
    if labels == {"document_index"}:
        return True
    lines = [l for l in chunk.page_content.splitlines() if l.strip()]
    return len(lines) >= 5 and sum(bool(LIST_LINE.search(l)) for l in lines) / len(lines) > 0.5


def normalise(text: str) -> str:
    return " ".join(text.replace("•", "·").split())   # Docling writes the bullet in "1.1 • Title" as "·"


def main() -> None:
    rules = load_rules_module()

    # ---------- method 1: the rules from script 1 (they need the whole book to count repetitions) ----------
    t = time.time()
    pages = PyPDFLoader(PDF_PATH).load()
    clean, dropped, boilerplate = rules.clean_pages(pages)
    rule_pages = {p.metadata["page"] + 1 for p in clean if in_ranges(p.metadata["page"] + 1)}
    rule_chunks = [c for c in rules.chunk(clean) if in_ranges(c.metadata["page"] + 1)]
    rule_seconds = time.time() - t

    # The actual header/footer lines on our pages, used to check what each method leaves behind.
    footers = {normalise(l) for p in pages if in_ranges(p.metadata["page"] + 1)
               for l in rules.lines_of(p.page_content) if rules.shape(l) in boilerplate}
    front_matter = {i + 1 for i, why in dropped.items() if why.startswith("front matter")}

    # ---------- method 2: Docling ----------
    t = time.time()
    raw = docling_chunks()
    dl_chunks = [c for c in raw if not is_toc_or_index(c)]
    dl_seconds = time.time() - t
    dl_pages = {p for c in dl_chunks for p in chunk_info(c)[1]}

    def footer_chunks(chunks):
        # Front-matter pages are skipped here (they have their own row): the copyright page
        # QUOTES "Access for free at openstax.org" as part of the licence text.
        return sum(any(f in normalise(c.page_content) for f in footers) for c in chunks
                   if not (pages_of(c) <= front_matter))

    def pages_of(c):
        return chunk_info(c)[1] if "dl_meta" in c.metadata else {c.metadata["page"] + 1}

    spanning = sum(len(chunk_info(c)[1]) > 1 for c in dl_chunks)
    print(f"\nPages compared: {' and '.join(f'{a}-{b}' for a, b in PAGE_RANGES)}")
    print(f"{'':44}{'rules (script 1)':>18}{'Docling':>12}")
    rows = [
        ("time (s)", f"{rule_seconds:.0f}", f"{dl_seconds:.0f}"),
        ("chunks", len(rule_chunks), len(dl_chunks)),
        ("chunks still containing a header/footer", footer_chunks(rule_chunks), footer_chunks(dl_chunks)),
        ("front-matter pages kept (cover, title...)", len(rule_pages & front_matter), len(dl_pages & front_matter)),
        ("TOC/index chunks removed by our filter", "-", len(raw) - len(dl_chunks)),
        ("chunks joining text across a page break", 0, spanning),
    ]
    for name, a, b in rows:
        print(f"  {name:42}{a!s:>18}{b!s:>12}")

    kept_front = sorted(dl_pages & front_matter)
    if kept_front:
        print(f"\nFront-matter pages Docling kept: {kept_front}. For example:")
        sample = next(c for c in dl_chunks if chunk_info(c)[1] & front_matter)
        print("   " + " ".join(sample.page_content.split())[:160] + "...")

    example = next((c for c in dl_chunks if len(chunk_info(c)[1]) > 1), None)
    if example:
        print(f"\nA Docling chunk that crosses a page break (pages {sorted(chunk_info(example)[1])}), no footer in it:")
        print("   " + " ".join(example.page_content.split())[:400] + "...")


if __name__ == "__main__":
    main()
