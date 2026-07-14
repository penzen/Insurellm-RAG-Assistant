# Insurellm RAG Assistant

A Retrieval-Augmented Generation application that allows users to ask questions about a fictional company called **Insurellm**.
The assistant retrieves relevant information from a local company knowledge base and uses an LLM to generate answers grounded in the provided context.



---

## Project Overview

This project demonstrates how to build a basic RAG system using:

* Markdown documents as a knowledge base
* Document loading and text chunking
* Vector embeddings
* Chroma vector database
* Semantic retrieval
* LLM-based answer generation
* Gradio chat interface

The knowledge base contains information about the company, employees, contracts, products, careers, culture, and general company details.

Instead of relying only on the LLM's internal knowledge, the app retrieves relevant document chunks from the vector database and provides them to the LLM as context before generating an answer.

---

## What This Project Does

The assistant can answer questions such as:

* What does Insurellm do?
* What products does the company offer?
* What is the company culture like?
* What career opportunities are available?
* What information is available about employees?
* What contract-related information is stored in the knowledge base?

The system also displays the retrieved context so the user can see which documents were used to generate the answer.

---

## RAG Architecture

The project follows a standard RAG pipeline:

```text
Knowledge Base Documents
        ↓
Load Markdown Files
        ↓
Split Documents into Chunks
        ↓
Create Vector Embeddings
        ↓
Store Embeddings in Chroma
        ↓
User Asks a Question
        ↓
Retrieve Relevant Chunks
        ↓
Send Context + Question to LLM
        ↓
Generate Final Answer
```

---

## Key Concepts Learned

### 1. Document Loading

The project loads `.md` files from the `knowledge-base` directory.

Each folder inside the knowledge base represents a document type, such as:

```text
knowledge-base/
├── company/
├── contracts/
├── employees/
└── products/
```

Each document is loaded and metadata is added so the system knows where the information came from.

---

### 2. Text Chunking

Large documents are split into smaller pieces called chunks.

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)
```

Chunking is important because LLMs and embedding models work better when documents are broken into meaningful sections.

The overlap helps prevent important information from being lost between chunk boundaries.

---

### 3. Vector Embeddings

Each chunk is converted into a numerical vector using an embedding model.

Example:

```text
"Employees receive company benefits"
        ↓
[0.12, -0.44, 0.91, ...]
```

These vectors represent the semantic meaning of the text.

Similar meanings should produce similar vectors.

---

### 4. Chroma Vector Database

The project uses Chroma to store document embeddings.

Chroma allows the app to search by meaning instead of simple keyword matching.

For example, a user might ask:

```text
What benefits do workers get?
```

Even if the document says:

```text
Employees receive benefits...
```

the retriever can still find the relevant chunk because the meanings are similar.

---

### 5. Retriever

The vector database is converted into a retriever:

```python
retriever = vectorstore.as_retriever()
```

The retriever searches the vector database and returns the most relevant document chunks for a user question.

---

### 6. Prompt Augmentation

The retrieved chunks are joined together into a context string.

```python
context = "\n\n".join(doc.page_content for doc in docs)
```

That context is inserted into a system prompt and sent to the LLM together with the user's question.

This is the "Augmented" part of Retrieval-Augmented Generation.

---

### 7. LLM Response Generation

The LLM receives:

* A system message containing the retrieved context
* The user's question
* Previous chat history

It then generates an answer based on the retrieved company information.

---

### 8. Gradio User Interface

The project includes a Gradio web interface with:

* A chatbot panel
* A user input box
* A retrieved context panel

This makes it easy to test the RAG system interactively.

---

## Project Structure

```text
WEEK_5/
├── knowledge-base/
│   ├── company/
│   ├── contracts/
│   ├── employees/
│   └── products/
│
├── vector_db/
│
├── ingest.py
├── answer.py
├── app.py
├── .env
└── README.md
```

---

## File Explanation

### `ingest.py`

This script prepares the knowledge base for retrieval.

It performs the ingestion pipeline:

```text
load documents → split into chunks → create embeddings → store in Chroma
```

Main responsibilities:

* Load markdown files from `knowledge-base`
* Add metadata to documents
* Split documents into chunks
* Create embeddings
* Save vectors to the local Chroma database

Run this file first before starting the app.

---

### `answer.py`

This file contains the main RAG logic.

It performs the query-time pipeline:

```text
user question → retrieve relevant chunks → build context → send to LLM → return answer
```

Main responsibilities:

* Load the Chroma vector database
* Create a retriever
* Retrieve relevant context documents
* Format the system prompt
* Call the LLM
* Return the final answer and retrieved context

---

### `app.py`

This file creates the Gradio interface.

Main responsibilities:

* Display the chatbot
* Accept user questions
* Send questions to the RAG pipeline
* Show the assistant answer
* Display retrieved context documents

---

## Installation

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install langchain langchain-chroma langchain-huggingface langchain-openai langchain-text-splitters chromadb gradio python-dotenv sentence-transformers
```

If using `uv`, install dependencies with:

```bash
uv pip install langchain langchain-chroma langchain-huggingface langchain-openai langchain-text-splitters chromadb gradio python-dotenv sentence-transformers
```

---

## Environment Variables

Create a `.env` file in the project root.

If using OpenAI for the chat model, add:

```text
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit the `.env` file to GitHub.

---

## How to Run

### 1. Ingest the documents

Run:

```bash
uv run ingest.py
```

or:

```bash
python ingest.py
```

This creates the local Chroma vector database.

Expected output:

```text
There are X vectors with 384 dimensions in the vector store
Ingestion complete
```

---

### 2. Start the Gradio app

Run:

```bash
uv run app.py
```

or:

```bash
python app.py
```

The app will open in the browser.

---

## Important Note About Embeddings

The same embedding model must be used during both ingestion and retrieval.

For example, if `ingest.py` creates vectors using:

```python
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

then `answer.py` should also use:

```python
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

If different embedding models are used, the vector dimensions may not match and retrieval can fail.

---

## Example RAG Flow

User asks:

```text
What does Insurellm do?
```

The app then:

1. Converts the question into an embedding
2. Searches Chroma for similar document chunks
3. Retrieves relevant company information
4. Adds the retrieved chunks to the prompt
5. Sends the prompt to the LLM
6. Returns a grounded answer to the user

---

## Technologies Used

* Python
* LangChain
* Chroma
* Hugging Face Embeddings
* OpenAI Chat Model
* Gradio
* dotenv
* Markdown knowledge base

---

## What I Learned

Through this project, I learned:

* How RAG systems work
* The difference between LLMs and embedding models
* How text is converted into vector embeddings
* How vector databases store and retrieve semantic information
* How chunk size and chunk overlap affect retrieval quality
* How Chroma stores local embeddings
* How a retriever searches for relevant context
* How to combine retrieved context with a user question
* How to build a simple Gradio interface for an LLM app
* Why using the same embedding model during ingestion and retrieval is important
* How to structure a basic LLM application with separate ingestion, retrieval, and UI files

---

## Future Improvements

Possible improvements for this project:

* Add better error handling
* Add support for PDF files
* Add document upload from the UI
* Add conversational query rewriting
* Add evaluation for retrieval quality
* Add Docker support
* Deploy the app to AWS
* Use a production vector database such as Qdrant, Pinecone, OpenSearch, or pgvector
* Add observability with LangSmith or LangFuse

---

## Summary

This project is a beginner-friendly but practical implementation of a RAG assistant.

It shows how company documents can be turned into searchable vector embeddings and used as context for an LLM-powered chatbot.

The main goal is to understand the full RAG workflow:

```text
documents → chunks → embeddings → vector database → retrieval → prompt → LLM answer
```

This is one of the core patterns used in modern LLM applications.
