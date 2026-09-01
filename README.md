# Helpbee

**Master's thesis project: "HELPBEE: Leveraging AI and Retrieval-Augmented
Generation (RAG) for Enhanced Customer Support Efficiency"**

Helpbee is a local, retrieval-augmented generation (RAG) chatbot that answers
customer FAQ questions grounded in a source document, using a locally hosted
LLM (Mistral, served via Ollama) instead of a cloud API. It was built to
study whether retrieval-augmented generation measurably improves customer
support answer quality over a plain LLM baseline, and to test that claim
with a proper statistical evaluation rather than anecdotal comparison.

## What it does

- Ingests a customer FAQ PDF and turns it into a searchable knowledge base
- Retrieves the most relevant question-answer pairs for a user's query using
  semantic vector search
- Generates a natural-language answer grounded in the retrieved context
- Serves the chatbot through a Streamlit chat UI
- Includes a statistical evaluation harness that compares the RAG pipeline
  against a non-RAG (plain LLM) baseline on faithfulness, relevance, context
  precision, and response time

## Architecture

**Ingestion pipeline (run once, or whenever the source document changes):**

1. `src/pdf_processor.py` - extracts raw text from the FAQ PDF with `pdfplumber`
2. `src/chunker.py` - cleans the text and splits it into individual
   question-answer chunks using the PDF's own numbered structure (regex-based,
   not fixed-size chunking)
3. `src/vector_store.py` - embeds each chunk with
   `sentence-transformers/multi-qa-mpnet-base-dot-v1` and writes the vectors
   into a FAISS index

**Query pipeline (per user request):**

4. `src/query_handler.py`:
   - embeds the user's query with the same embedding model
   - retrieves the top-k nearest chunks from the FAISS index, filtered by a
     distance threshold
   - applies a keyword-overlap filter as a second precision pass
   - builds a prompt from the filtered context and the user's question
   - generates the final answer with Mistral via Ollama (fully local, no
     external API calls)
5. `src/app.py` - Streamlit chat interface, launched via `main.py`

```
FAQ PDF -> pdfplumber -> clean & chunk (Q&A pairs) -> sentence-transformers
embeddings -> FAISS index
                                                            |
User query -> embed -> FAISS top-k search -> keyword filter -> prompt ->
Mistral (Ollama) -> answer -> Streamlit UI
```

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python |
| PDF extraction | pdfplumber |
| Chunking | Custom regex-based Q&A splitting |
| Embeddings | sentence-transformers (`multi-qa-mpnet-base-dot-v1`), PyTorch |
| Vector search | FAISS (flat index) |
| LLM | Mistral 7B, served locally via Ollama |
| UI | Streamlit |
| Evaluation | pandas, scikit-learn, rouge-score, SciPy-based statistical tests |

No LangChain or other orchestration framework is used - retrieval, filtering,
prompt construction, and generation are implemented directly.

## Getting started

### Prerequisites

- Python 3.12
- [Ollama](https://ollama.com) installed and running locally

### Setup

```bash
# 1. Install Ollama and pull the model the app uses
ollama pull mistral

# 2. Install Python dependencies
pip install -r requirements.txt
pip install ollama

# 3. (Optional) Rebuild the knowledge base - only needed if you change the
#    source PDF or chunking logic. The repo already ships a prebuilt index
#    and chunk files under data/ and vector_store/.
python src/pdf_processor.py
python src/chunker.py
python src/vector_store.py

# 4. Launch the chatbot
python main.py
```

`main.py` runs `streamlit run src/app.py`, which opens the chat UI in your
browser.

## Configuration

All paths and model names are centralized in `src/config.py`:

- `FAQ_PDF_PATH` - source FAQ PDF
- `EXTRACT_TEXT_DATA_PATH`, `CHUNK_DATA_PATH`, `VECTOR_STORE_PATH` - pipeline artifact locations
- `EMBEDDING_MODEL` - sentence-transformers model used for embeddings
- `LANGUAGE_MODEL` / `MODEL_NAME` - Ollama platform and model name

## Evaluation

`evaluation/` contains a separate harness that runs the RAG pipeline and a
non-RAG baseline (plain Mistral, no retrieval) over the same set of 10
representative FAQ questions across 3 iterations, then compares them on:

- **Faithfulness (cosine similarity)** - how closely the generated answer
  aligns with the retrieved context
- **Faithfulness (ROUGE-L)** - lexical overlap between the answer and the
  retrieved context
- **Answer relevance** - how closely the answer matches the query itself
- **Context precision** - how relevant the retrieved chunks are to the
  generated answer
- **Response time**

Before comparing groups, each metric's distribution was checked for
normality (Shapiro-Wilk) and equal variance (Levene's test), then compared
with a paired t-test where normality held, or a Wilcoxon signed-rank test
otherwise.

**Results:**

| Metric | RAG mean | Non-RAG mean | Test | p-value | Significant |
|---|---|---|---|---|---|
| Faithfulness (cosine) | 0.83 | 0.72 | paired t-test | 0.0003 | Yes |
| Faithfulness (ROUGE-L) | 0.60 | 0.20 | Wilcoxon | 0.0076 | Yes |
| Context precision | 0.69 | 0.62 | paired t-test | 0.0009 | Yes |
| Answer relevance | 0.72 | 0.81 | Wilcoxon | 0.47 | No |
| Response time (s) | 21.99 | 18.06 | paired t-test | 0.27 | No |

RAG produced significantly more grounded, source-faithful answers than the
non-RAG baseline. Answer relevance did not improve significantly with RAG -
a plain LLM can sound just as on-topic even when its answer isn't grounded
in any real source, which is exactly the failure mode RAG is meant to guard
against. Full per-question output is in `evaluation/findings/`.

## Known limitations

- Evaluated on a small test set (10 questions) - useful for a controlled
  statistical comparison, not a large-scale benchmark
- No classic IR retrieval metrics (Precision@k, Recall@k, MRR) - evaluation
  focuses on generation-quality proxies rather than isolated retrieval
  quality against labeled relevance judgments
- No LLM-as-judge scoring - faithfulness and relevance are measured via
  embedding cosine similarity and ROUGE, not semantic correctness grading
- English-only, single-document knowledge base
- No source citations returned alongside answers
- Faithfulness is measured offline only, not enforced as a live guardrail
  in the running app
- `ollama` Python package is required at runtime but not yet pinned in
  `requirements.txt`

## Possible extensions

- Hybrid retrieval (dense FAISS + BM25 sparse search)
- Cross-encoder re-ranking after initial retrieval
- Query rewriting/expansion before embedding
- Domain-specific fine-tuning of the embedding model on the FAQ corpus
- Multilingual embedding model for non-English source documents
- Metadata-aware vector store for filtering by document type/date/source
- LLM-as-judge evaluation alongside the existing similarity-based metrics
- Live faithfulness guardrail that rejects or flags low-confidence answers

## Project structure

```
.
|-- main.py                    Entry point - launches the Streamlit app
|-- src/
|   |-- config.py              Paths and model configuration
|   |-- pdf_processor.py       PDF text extraction
|   |-- chunker.py             Text cleaning and Q&A chunking
|   |-- vector_store.py        Embedding and FAISS index creation
|   |-- query_handler.py       Retrieval, filtering, and answer generation
|   `-- app.py                 Streamlit chat UI
|-- data/
|   |-- raw/                   Source FAQ PDF
|   |-- extracted_text/        Extracted plain text
|   `-- chunk_data/            Individual Q&A chunks
|-- vector_store/              Prebuilt FAISS index
|-- evaluation/                RAG vs. non-RAG evaluation harness
|   |-- config.py              Evaluation-side configuration
|   |-- evaluate_rag.py        RAG pipeline evaluation run
|   |-- evaluate_baseline.py   Non-RAG baseline evaluation run
|   |-- baseline_app.py        Standalone plain-Mistral demo (no retrieval)
|   |-- stats_tests.py         Paired significance testing
|   |-- notebooks/             Exploratory data analysis notebooks
|   |-- results/               Aggregated metrics, CSVs, and comparison plots
|   `-- findings/              Raw per-question output and normality/variance tests
`-- archive/                   Earlier prototype GUI, kept for reference
```

## License

MIT - see [LICENSE](LICENSE).
