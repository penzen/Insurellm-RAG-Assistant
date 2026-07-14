# Insurellm RAG Assistant

An advanced Retrieval-Augmented Generation (RAG) assistant for a fictional insurance technology company called **Insurellm**.

The application lets users ask natural-language questions about company documents, products, employees, contracts, careers, and company culture. It retrieves relevant information from a local knowledge base, improves retrieval with query rewriting and reranking, and generates grounded answers using an LLM.



---

## Project Overview

The goal of this project is to move beyond a basic chatbot and build a more realistic RAG system.

Instead of asking the LLM to answer from its own training data, the app first searches a company knowledge base, retrieves relevant document chunks, and gives those chunks to the LLM as context.

The project includes two RAG implementations:

1. **Basic RAG implementation** using LangChain, Hugging Face embeddings, Chroma, and recursive text splitting.
2. **Advanced RAG implementation** using LLM-assisted preprocessing, OpenAI embeddings, query rewriting, query expansion, reranking, and an evaluation dashboard.

The current Gradio app uses the **advanced implementation**.

---

## Knowledge Base

The knowledge base is stored as Markdown files under `knowledge-base/`.

It contains documents about:

- Company overview
- Company culture
- Careers
- Insurance AI products
- Employee profiles
- Customer contracts
- Product/customer relationships

Example structure:

```text
knowledge-base/
├── company/
├── contracts/
├── employees/
└── products/
```

This knowledge base acts as the source of truth for the assistant.

---

## Core RAG Flow

The main RAG workflow is:

```text
Markdown knowledge base
        ↓
Load documents
        ↓
Split or preprocess documents into chunks
        ↓
Create vector embeddings
        ↓
Store embeddings in Chroma
        ↓
User asks a question
        ↓
Rewrite and expand the query
        ↓
Retrieve relevant chunks
        ↓
Rerank retrieved chunks
        ↓
Send best context + question to the LLM
        ↓
Generate grounded answer
```

---

## Features

### Basic RAG

The basic implementation demonstrates the standard RAG pattern:

- Load Markdown documents from the knowledge base
- Split documents using `RecursiveCharacterTextSplitter`
- Create embeddings with `all-MiniLM-L6-v2`
- Store vectors in Chroma
- Retrieve relevant context with a LangChain retriever
- Generate answers with an OpenAI chat model

### Advanced RAG

The advanced implementation adds production-style RAG improvements:

- LLM-assisted document chunking
- Structured chunk metadata using Pydantic
- Headline + summary + original text for each chunk
- OpenAI `text-embedding-3-large` embeddings
- Chroma `PersistentClient` vector storage
- Query rewriting
- Query expansion
- LLM-based reranking
- Final top-k context selection
- Gradio interface showing retrieved context

### Evaluation

The project includes an evaluation system for testing retrieval and answer quality.

It uses a JSONL test set with multiple question categories:

- `direct_fact`
- `temporal`
- `spanning`
- `comparative`
- `numerical`
- `relationship`
- `holistic`

The evaluation system measures:

- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Keyword coverage
- Answer accuracy
- Answer completeness
- Answer relevance

The answer evaluation uses an LLM-as-a-judge approach.

---

## Project Structure

```text
Insurellm-RAG-Assistant/
├── app.py
├── evaluator.py
├── README.md
├── .gitignore
│
├── implementation/
│   ├── ingest.py
│   └── answer.py
│
├── pro_implementation/
│   ├── ingest.py
│   └── answer.py
│
├── evaluation/
│   ├── eval.py
│   ├── test.py
│   └── tests.jsonl
│
└── knowledge-base/
    ├── company/
    ├── contracts/
    ├── employees/
    └── products/
```

Generated folders such as `vector_db/`, `preprocessed_db/`, `.venv/`, and `__pycache__/` should not be committed to GitHub.

---

## File Breakdown

### `app.py`

Launches the main Gradio chat interface.

It imports the advanced RAG pipeline from:

```python
from pro_implementation.answer import answer_question
```

The UI contains:

- A chatbot panel
- A user question box
- A retrieved context panel

The retrieved context panel is useful for debugging because it shows which knowledge-base chunks were used to answer the question.

---

### `implementation/ingest.py`

Creates the basic vector database.

Main steps:

1. Load Markdown documents from `knowledge-base/`
2. Split documents into chunks
3. Create embeddings with Hugging Face `all-MiniLM-L6-v2`
4. Store embeddings in Chroma

This version is useful for understanding the basic RAG pipeline.

---

### `implementation/answer.py`

Implements the basic RAG answering flow.

Main steps:

1. Load the Chroma vector store
2. Convert the vector store into a retriever
3. Retrieve relevant chunks for a user question
4. Build a system prompt with the retrieved context
5. Call the LLM
6. Return the generated answer and context

---

### `pro_implementation/ingest.py`

Creates the advanced preprocessed vector database.

Instead of simple character-based chunking, this file uses an LLM to split each document into richer chunks.

Each chunk has:

```python
headline: str
summary: str
original_text: str
```

The final chunk text combines all three fields:

```text
headline

summary

original_text
```

This improves retrieval because the vector database stores both the original evidence and an LLM-generated description of what the chunk is about.

This script stores embeddings in `preprocessed_db/`.

---

### `pro_implementation/answer.py`

Implements the advanced RAG answering pipeline.

Main features:

#### Query Rewriting

The user's question is rewritten into a clearer knowledge-base search query.

Example:

```text
Original: What about the second one?
Rewritten: What are the details of Insurellm's second product?
```

This helps retrieval work better in conversations.

#### Query Expansion

The system retrieves chunks for both:

- the original user question
- the rewritten search query

The results are merged to improve coverage.

#### Reranking

The first retrieval step collects a broader set of chunks. Then an LLM reranker sorts those chunks by relevance to the user's question.

The final answer uses only the top-ranked chunks.

#### Answer Generation

The final prompt includes:

- system instructions
- retrieved context extracts
- user question

The model is instructed to answer accurately, completely, and only when the answer is supported by the knowledge base.

---

### `evaluation/tests.jsonl`

A JSONL evaluation dataset.

Each line contains one test case with:

```json
{
  "question": "Who won the prestigious IIOTY award in 2023?",
  "keywords": ["Maxine", "Thompson", "IIOTY"],
  "reference_answer": "Maxine Thompson won the prestigious Insurellm Innovator of the Year award in 2023.",
  "category": "direct_fact"
}
```

The test categories help evaluate different RAG capabilities:

| Category | Meaning |
|---|---|
| `direct_fact` | Simple factual questions answered from one clear source |
| `temporal` | Questions involving time, dates, or sequence |
| `spanning` | Questions requiring multiple chunks or documents |
| `comparative` | Questions comparing two or more entities |
| `numerical` | Questions involving numbers, counts, prices, or amounts |
| `relationship` | Questions about connections between people, products, contracts, or teams |
| `holistic` | Broad summary or big-picture questions |

---

### `evaluation/test.py`

Loads the JSONL test cases into Pydantic models.

This gives a clean structure for each test question:

```python
class TestQuestion(BaseModel):
    question: str
    keywords: list[str]
    reference_answer: str
    category: str
```

---

### `evaluation/eval.py`

Runs retrieval and answer evaluations.

Retrieval evaluation checks whether the retrieved chunks contain the expected keywords.

Metrics:

- **MRR**: how high the first useful chunk appears
- **nDCG**: whether useful chunks are ranked near the top
- **Keyword coverage**: how many expected keywords appear in retrieved context

Answer evaluation uses an LLM judge to score:

- **Accuracy**
- **Completeness**
- **Relevance**

---

### `evaluator.py`

Launches a Gradio evaluation dashboard.

The dashboard can run:

- Retrieval evaluation
- Answer evaluation

It displays:

- Average MRR
- Average nDCG
- Average keyword coverage
- Average answer accuracy
- Average answer completeness
- Average answer relevance
- Category-level bar charts

---

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install openai python-dotenv pydantic chromadb tqdm litellm numpy scikit-learn plotly gradio pandas tenacity langchain langchain-chroma langchain-huggingface langchain-openai langchain-text-splitters sentence-transformers
```

If using `uv`:

```bash
uv pip install openai python-dotenv pydantic chromadb tqdm litellm numpy scikit-learn plotly gradio pandas tenacity langchain langchain-chroma langchain-huggingface langchain-openai langchain-text-splitters sentence-transformers
```

---

## Environment Variables

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit `.env` to GitHub.

---

## How to Run

### 1. Create the advanced vector database

```bash
uv run pro_implementation/ingest.py
```

or:

```bash
python pro_implementation/ingest.py
```

This creates `preprocessed_db/` locally.

### 2. Run the chat app

```bash
uv run app.py
```

or:

```bash
python app.py
```

The Gradio app opens in the browser.

### 3. Run the evaluation dashboard

```bash
uv run evaluator.py
```

or:

```bash
python evaluator.py
```

### 4. Run a single CLI evaluation test

Depending on your Python path setup, run from the project root:

```bash
python -m evaluation.eval 0
```

or from inside the `evaluation/` directory:

```bash
python eval.py 0
```

---

## What I Learned

This project helped me understand and implement:

- How RAG systems work end to end
- The difference between LLMs and embedding models
- How embeddings represent text as vectors
- How Chroma stores and searches vectors
- How chunk size and overlap affect retrieval
- Why retrieval quality matters before answer generation
- How to build a basic LangChain RAG app
- How to build a more advanced RAG pipeline manually
- How Pydantic structured outputs can guide LLM preprocessing
- How LLM-generated headlines and summaries can improve retrieval
- How query rewriting improves conversational search
- How query expansion improves context coverage
- How reranking improves the order of retrieved chunks
- How to evaluate retrieval with MRR, nDCG, and keyword coverage
- How to evaluate answers with LLM-as-a-judge
- How to build a Gradio app and a Gradio evaluation dashboard

---

## Important Engineering Notes

### Same embedding model must be used for indexing and querying

If the database is created with OpenAI `text-embedding-3-large`, the query embedding must also use `text-embedding-3-large`.

If the database is created with Hugging Face `all-MiniLM-L6-v2`, querying must use the same model.

Mixing embedding models can cause dimension mismatches or poor retrieval.

### Generated vector databases are not committed

The folders `vector_db/` and `preprocessed_db/` are generated artifacts. They can be recreated by running the ingestion scripts, so they should be ignored by Git.

### The app depends on ingestion

The chat app expects the advanced Chroma database to exist. Run `pro_implementation/ingest.py` before running `app.py`.

---

## Recommended `.gitignore`

```gitignore
.env
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
notebook/
vector_db/
preprocessed_db/
.DS_Store
```

---

## Future Improvements

Possible next steps:

- Add source citations directly into the final answer
- Replace LLM reranking with a faster dedicated reranker model
- Add hybrid retrieval using vector search plus keyword/BM25 search
- Add metadata filters by document type
- Add document upload from the UI
- Add support for PDFs
- Add Docker support
- Add deployment to AWS
- Add LangSmith or LangFuse tracing
- Store evaluation results over time
- Add CI checks for retrieval quality before deployment

---

## Summary

This project demonstrates a complete RAG learning path:

```text
basic RAG
    ↓
advanced preprocessing
    ↓
query rewriting and expansion
    ↓
reranking
    ↓
answer generation
    ↓
retrieval and answer evaluation
```

It shows how a company knowledge base can be converted into a searchable vector database and used to power a grounded LLM assistant.

The most important lesson is that a RAG system should not only answer questions — it should also be evaluated to verify that it retrieves the right evidence and generates accurate, relevant, and complete answers.
