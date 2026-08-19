# LLM RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant information from a technical document using semantic search and generates context-grounded answers with an LLM.

## Project Overview

This project implements a custom Retrieval-Augmented Generation pipeline to understand the core components of a RAG system rather than treating RAG as a black-box workflow.

The current knowledge source is the research paper:

**Attention Is All You Need**

The system processes the document, divides it into smaller chunks, generates semantic embeddings, stores the knowledge base, and uses FAISS for similarity-based retrieval.

For every user query, the most relevant document chunks are retrieved and supplied to the LLM as context before generating the final answer.

---

## What is RAG?

Retrieval-Augmented Generation combines information retrieval with Large Language Models.

Instead of relying only on the LLM's internal knowledge, RAG retrieves relevant information from an external knowledge source and provides it to the model as context.

The basic workflow is:

    User Query
         |
         v
    Query Embedding
         |
         v
    Vector Similarity Search
         |
         v
    Relevant Document Chunks
         |
         v
    RAG Prompt
         |
         v
    LLM
         |
         v
    Grounded Answer

This approach allows an LLM to answer questions using information retrieved from a specific knowledge source.

---

## Architecture

    Knowledge Source
           |
           v
          PDF
           |
           v
    Text Extraction
           |
           v
        Chunking
           |
           v
    Sentence Transformer
       Embeddings
           |
           +----------------------+
           |                      |
           v                      v
    knowledge_base.pkl       FAISS Index
    chunks + embeddings      vector search
           |                      |
           +----------+-----------+
                      |
                      v
                 User Query
                      |
                      v
              Query Embedding
                      |
                      v
                 FAISS Search
                      |
                      v
                 Top-K Chunks
                      |
                      v
                  RAG Prompt
                      |
                      v
                    LLM API
                      |
                      v
                   Answer

---

## Pipeline

### 1. Document Extraction

The PDF document is processed and its text is extracted using `pypdf`.

### 2. Text Chunking

The extracted text is divided into smaller overlapping chunks using LangChain's text splitting utilities.

The current implementation uses:

- Chunk size: 500 characters
- Chunk overlap: 50 characters
- Total chunks in the current knowledge source: 89

Chunk overlap helps preserve contextual information between neighboring chunks.

### 3. Embedding Generation

Each document chunk is converted into a semantic vector using the Sentence Transformers model:

    all-MiniLM-L6-v2

The generated embeddings have 384 dimensions.

For the current knowledge source:

    89 chunks × 384 dimensions

### 4. Persistent Knowledge Base

The chunks and their embeddings are stored locally in:

    data/knowledge_base.pkl

This allows the application to reuse the generated knowledge base instead of processing and embedding the document every time.

The `data/` directory is excluded from version control because these are generated artifacts.

### 5. FAISS Vector Search

FAISS is used to perform efficient similarity search over the document embeddings.

The implementation uses:

    FAISS IndexFlatIP

The embeddings are L2-normalized before indexing. For normalized vectors, Inner Product search corresponds to cosine similarity.

For every user query, the system retrieves the top 3 most relevant chunks.

### 6. Persistent FAISS Index

The FAISS index is saved locally as:

    data/faiss_index.bin

On subsequent application runs, the existing FAISS index is loaded from disk instead of being rebuilt.

This separates the one-time knowledge-base/index creation process from repeated query-time retrieval.

### 7. Query Processing

When the user enters a question:

    User Query
         |
         v
    Query Embedding
         |
         v
    Normalization
         |
         v
    FAISS Similarity Search
         |
         v
    Top 3 Relevant Chunks

The retrieved chunks are then passed to the generation layer.

### 8. RAG Prompt Construction

The retrieved chunks are inserted into a prompt along with the user's question.

The prompt instructs the LLM to answer using the supplied context.

### 9. Answer Generation

The final answer is generated using the OpenAI API.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Programming language |
| pypdf | PDF text extraction |
| LangChain Text Splitters | Text chunking |
| Sentence Transformers | Semantic embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| NumPy | Numerical and vector operations |
| Scikit-learn | Similarity experiments and development |
| OpenAI API | LLM-based answer generation |
| python-dotenv | Environment variable management |
| Conda | Environment management |

---

## Project Structure

    LLM-RAG-Assistant/
    |
    +-- app/
    |   +-- extract_text.py
    |   +-- chunk_text.py
    |   +-- embeddings.py
    |   +-- retrieve.py
    |   +-- generate_answer.py
    |   +-- faiss_test.py
    |   +-- test_api.py
    |
    +-- documents/
    |   +-- attention_is_all_you_need.pdf
    |
    +-- data/
    |   +-- knowledge_base.pkl
    |   +-- faiss_index.bin
    |
    +-- notebooks/
    |
    +-- .gitignore
    +-- requirements.txt
    +-- README.md

Note: The `data/` directory contains generated files and is excluded from Git version control.

The `.env` file containing the API key is also excluded from version control.

---

## Setup

### 1. Clone the Repository

    git clone https://github.com/ash21agrawal/LLM-RAG-Assistant.git
    cd LLM-RAG-Assistant

### 2. Create the Conda Environment

    conda create -n llm-rag python=3.11
    conda activate llm-rag

### 3. Install Dependencies

    python -m pip install -r requirements.txt

### 4. Configure the OpenAI API Key

Create:

    app/.env

Add:

    OPENAI_API_KEY=your_api_key_here

Never commit the `.env` file to GitHub.

---

## Running the Project

The current implementation uses the PDF placed inside the `documents/` directory as the knowledge source.

### First Run

Run:

    python app/retrieve.py

On the first run, the system creates the knowledge base and FAISS index.

The process is:

    PDF
     |
     v
    Text Extraction
     |
     v
    Chunking
     |
     v
    Embedding Generation
     |
     v
    knowledge_base.pkl
     |
     v
    FAISS Index
     |
     v
    faiss_index.bin

### Subsequent Runs

When the knowledge base and FAISS index already exist, the application loads them from disk.

This avoids repeating document processing and FAISS index construction.

### Run the Complete RAG Pipeline

    python app/generate_answer.py

The application then accepts interactive queries:

    Enter your query (type 'exit' to quit):

---

## Example

### Query

    What is multi-head attention?

### Retrieval

The system generates an embedding for the query and searches the FAISS index.

The top 3 semantically relevant chunks are retrieved from the Transformer paper.

### Generated Answer

The LLM receives the retrieved chunks as context and generates an answer based on that context.

Example:

    Multi-head attention is a mechanism that runs several attention
    layers in parallel. Each head uses learned query, key, and value
    projections to compute attention, and the results are combined
    to produce the output.

---

## Retrieval Configuration

| Parameter | Value |
|---|---|
| Embedding Model | `all-MiniLM-L6-v2` |
| Embedding Dimension | 384 |
| Number of Chunks | 89 |
| Chunk Size | 500 characters |
| Chunk Overlap | 50 characters |
| Vector Index | FAISS `IndexFlatIP` |
| Similarity | Cosine similarity using normalized Inner Product |
| Top-K Retrieval | 3 |

---

## Persistence

One of the important design decisions in this project is that the generated knowledge base and vector index are persisted locally.

The first run performs:

    Document
       |
       v
    Chunking
       |
       v
    Embeddings
       |
       v
    Save Knowledge Base
       |
       v
    Build FAISS Index
       |
       v
    Save FAISS Index

Later runs perform:

    knowledge_base.pkl
          |
          v
        Chunks

    faiss_index.bin
          |
          v
      FAISS Index

Both are loaded directly without rebuilding the document representation.

---

## Current Capabilities

- Extract text from PDF documents
- Split documents into overlapping chunks
- Generate semantic embeddings
- Store chunks and embeddings persistently
- Build a FAISS vector index
- Persist the FAISS index to disk
- Perform semantic similarity search
- Retrieve top-K relevant chunks
- Construct a context-grounded RAG prompt
- Generate answers using an LLM
- Interactively query the knowledge source through the terminal

---

## Current Limitations

This is a Phase-I implementation focused on understanding the core RAG pipeline.

Current limitations include:

- Single knowledge document
- Terminal-based interaction
- No graphical user interface
- No user authentication
- No multi-user document isolation
- No document management interface
- No conversational memory

---

## Future Scope — Phase II

The next phase will transform the current RAG engine into a user-facing application.

Planned features include:

- Web-based frontend
- Chatbot interface
- Multiple document uploads
- PDF, DOCX and TXT support
- User-specific document collections
- Automatic document ingestion
- Persistent vector database
- Conversational chat history
- Document management
- Source-aware responses
- Cloud deployment

The goal is to allow users to upload their own documents and ask natural-language questions about the information contained within them.

---

## Knowledge Source

The current demonstration uses the research paper:

**Attention Is All You Need**

by Ashish Vaswani et al.

The paper is used as the knowledge source for demonstrating the RAG pipeline.

---

## Project Status

### Phase I — Core RAG Pipeline

**Complete**

    PDF
     |
     v
    Text Extraction
     |
     v
    Chunking
     |
     v
    Embeddings
     |
     v
    Persistent Knowledge Base
     |
     v
    FAISS Vector Search
     |
     v
    Top-K Retrieval
     |
     v
    RAG Prompt
     |
     v
    LLM
     |
     v
    Answer

### Phase II — Multi-Document RAG Application

**Planned**

The next phase will build a complete frontend application around the existing RAG engine.

---

## Key Learning Outcomes

This project was built to understand the internal components of a RAG system instead of relying entirely on a black-box RAG framework.

Key concepts implemented and explored:

- Document preprocessing
- Text chunking
- Chunk overlap
- Semantic embeddings
- Vector representations
- Cosine similarity
- FAISS indexing
- Top-K retrieval
- Persistent knowledge bases
- Persistent vector indexes
- RAG prompt construction
- Context-grounded generation
- Separation of retrieval and generation layers