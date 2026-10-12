# 17_problems: uncommon, not-so-obvious problems in agentic AI

Each subfolder holds one area. Each numbered script states one problem, shows it on real data, and shows a solution. Every script runs on its own from its own folder, with the repo's `.venv` active (unless the script says it needs its own venv).

## RAG

| # | Script | Problem | Solution in one line |
|---|---|---|---|
| 1 | `RAG/1_pdf_headers_footers_cleanup.py` | Title/front-matter pages, TOC, index, and running headers/footers get loaded, chunked and retrieved as if they were content. | Find header/footer lines by **repetition** at page edges (digits normalised so page numbers don't matter), drop front matter (before page numbering starts), drop list-like pages (TOC, index), then chunk. |
| 2 | `RAG/2_pdf_tables_to_rows.py` | A loader flattens tables into a stream of words: wrapped cells break apart, and rows lose their column names and caption once chunked. | Extract real tables with pdfplumber and turn each row into a self-contained document (`caption \| Column: value \| ...`). Take the prose from outside the table area so the jumbled version isn't indexed too. |
| 1b | `RAG/1b_pdf_cleanup_with_docling.py` | Same problem, solved with an open-source layout model (Docling) and compared with the rules. | `DoclingLoader` drops headers/footers by layout. Front matter and the TOC still need filtering; chunks keep sentences together across page breaks. |

### Backlog: problems to solve next

1. **Sentences split across page breaks:** a chunk boundary falls on a PDF page break, so half a sentence lands in each chunk.
2. **Tables flattened into text:** rows and columns turn into a jumble of numbers, so the model can't tell which value belongs to which column. *(solved: `RAG/2_pdf_tables_to_rows.py`)*
3. **Multi-column layouts:** text from two columns gets interleaved line by line.
4. **Duplicate and near-duplicate documents:** several versions of the same policy fill all the top-k slots and push out other relevant chunks.
5. **Stale versions:** an outdated document outranks the current one because it has more matching words.
6. **Chunks that lose their context:** a chunk like "It is covered after 24 months" doesn't say what "it" refers to, because the heading or subject was in an earlier chunk.
7. **Numbers, codes and IDs don't match by meaning:** queries such as "error E1043" or "policy P12345" fail with embedding-only search.
8. **Negation and exclusions:** "What is NOT covered?" retrieves the coverage clauses instead of the exclusions.
9. **Prompt injection hidden in documents:** a retrieved PDF or web page contains instructions such as "ignore previous instructions", and the agent follows them.
10. **Tool-call loops:** an agent keeps calling the same tool with the same arguments and never stops (until max turns or the token budget runs out).

### 1. PDF headers, footers and title pages

- Data: `9_general/rag/Principles-of-Data-Science.pdf` (OpenStax, 569 pages).
- Run: `cd 17_problems/RAG` then `python 1_pdf_headers_footers_cleanup.py`, or pass your own query: `python 1_pdf_headers_footers_cleanup.py "your query"`. No API key is needed, and it takes about 40–80 seconds, mostly for loading the PDF.
- What it shows on this book:
  - **Pages:** 569 loaded, 544 kept. Dropped: 6 front-matter pages, 9 TOC/index pages, and 10 pages that held nothing but a footer.
  - **Header/footer shapes:** 66 found automatically, e.g. `Access for free at openstax.org` on 281 pages, and one running footer per chapter, section and appendix.
  - **Chunks containing a header/footer line:** 536 noisy, 0 clean.
  - **Retrieval (BM25), noisy version:** "hallucinations" ranks an index page #1, and "R Studio" retrieves a page that contains only a footer. In the clean version both disappear.
- Why the rules look the way they do:
  - **Repetition alone is not enough.** Content labels such as `PYTHON CODE` or `EXAMPLE 3.2` also repeat at page edges. A repeated edge line is treated as a header/footer only if it has a page number at its start/end, or appears on at least 25% of pages.
  - **It must contain a letter.** Lines such as `9 4766.18` are table data that happen to sit at a page edge.
  - **Footers are removed first, short pages are dropped after.** Many short pages are the real last lines of a section.
- Tuning: the constants at the top of the script (`EDGE_TOP`, `EDGE_BOTTOM`, `MIN_REPEAT`, `FREQUENT_SHARE`, `LIST_PAGE_SHARE`, `MIN_CONTENT_CHARS`).
- Limits: it works on text lines. It does not handle multi-column layouts, scanned PDFs (those need OCR), or a sentence that is split across a page break; that last one is a good candidate for a later problem.

### 1b. The same problem with an open-source library (Docling)

`RAG/1b_pdf_cleanup_with_docling.py` loads the book with LangChain's `DoclingLoader`. Docling (IBM, MIT licence) runs a layout model that labels page headers and footers, so they never reach the chunks. The script runs **both** methods on the same pages and prints a comparison.

- **Install it in a separate venv.** Docling brings PyTorch and layout models, so don't install it into the shared `.venv`. The exact command is at the top of the script.
- **Version note:** `langchain-docling` 3.x needs `langchain-core` 1.x. Use **`langchain-docling==1.1.0`**, which works with this repo's `langchain-core==0.3.86` pin. I checked this against the full `requirements.txt` (without `deepeval`, as described in CLAUDE.md) with a dry-run install.
- **Run:** `python 1b_pdf_cleanup_with_docling.py` uses sample pages 1–30 and 560–569 (about 2 minutes). Add `--all` for the whole book.

Results on the sample pages (CPU, no OCR, no table model):

| | Rules (script 1) | Docling |
|---|---|---|
| Time | 27 s (whole PDF loaded) | 68 s for 40 pages |
| Chunks still containing a header/footer | 0 | 4 |
| Front-matter pages kept (cover, title, copyright, ads) | 0 | 4 |
| Chunks joining text across a page break | 0 | 12 |

The whole book (`--all`), same machine:

| | Rules (script 1) | Docling |
|---|---|---|
| Time | 29 s | 825 s (~14 min) |
| Chunks | 1,625 | 1,615 |
| Chunks still containing a header/footer | 0 | 7 |
| Front-matter pages kept | 0 | 4 |
| TOC/index chunks removed by the extra filter | – | 27 |
| Chunks joining text across a page break | 0 | 355 |

What this shows:
- **Docling misses some headers and footers.** On one Preface page it read the header `4 Preface` as a *section heading* and attached it to 3 chunks; one more stayed in a chunk spanning pages 17–19.
- **Docling keeps front matter.** Cover, title, copyright and ad pages look like ordinary text to a layout model.
- **Docling labels the index (`document_index`) but not the TOC.** Both still need the small list-page filter from script 1, which this script reuses.
- **Where Docling wins:** chunks follow the document's sections, carry the section heading, and keep sentences together across page breaks. Script 1 can't do that, because PyPDFLoader produces one document per page.
- **Speed:** about 1.7 s per page on a CPU, so the whole book takes roughly 15–20 minutes. A GPU build of PyTorch is much faster.

**Choosing between them:** the rules are fast, dependency-free and reproducible. Docling gives better-structured chunks. In practice, combine them: Docling for structure, plus the front-matter and TOC/index filters.

### 2. Tables flattened into text

- Run: `cd 17_problems/RAG` then `python 2_pdf_tables_to_rows.py`. That reads pages 20–110 in about 30 seconds. Add `--all` for the whole book (about 2 minutes), or pass a question: `python 2_pdf_tables_to_rows.py "your question"`. It uses `pdfplumber`, which is already pinned in `requirements.txt`. No API key is needed.
- **The problem, on the book's Table 1.1:** a plain loader produces `Not / Fall 2020 A 100 recommended / at all`. One cell ("Not recommended at all") ends up on three lines around the other cells of its row, and the caption only appears after the rows.
- **The solution:** each row becomes one document:
  `Table 1.1 Course Evaluation Records | Semester: Spring 2022 | Instructor: B | Class Size: 40 | Rating: Highly recommended`.
  Column names and caption are repeated in every row, so a row makes sense, and can be found, however it is chunked.
- **Results:**
  - Whole book: 117 data tables → 733 row documents, and every table's caption was found.
  - "What was the rating of the class with a class size of 40 in Spring 2022?": the plain version retrieves the page's prose; the table-aware version returns exactly that row.
  - "benefit of data security in a DBMS": the plain version returns jumbled column text; the table-aware version returns the Data security row with labelled Description and Benefit.
- **Details that matter:**
  - **Only tables with at least 2 columns and 3 rows count.** This book also draws boxes around notes and code, which pdfplumber reports as 1-column tables (it flags 367 pages, most of them boxes).
  - **The prose is taken with `page.outside_bbox(table.bbox)`,** so the scrambled table text isn't indexed a second time.
  - **BM25 needs a lower-casing tokenizer.** Its default `str.split()` wouldn't match "Rating" with "rating".
- **Limits:**
  - pdfplumber finds tables from ruling lines, so tables drawn without lines are missed (Docling's table model handles those, at a much higher cost).
  - A caption that wraps onto a second line is cut short.
  - Math symbols in headers (e.g. *z*) come out empty.
  - A table that continues onto the next page is treated as two tables. Its rows are still labelled, because both parts have header rows in this book.
