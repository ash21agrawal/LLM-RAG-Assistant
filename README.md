# LLM RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that retrieves relevant information from a technical document using semantic search and generates context-grounded answers with an LLM.

## 📌 Project Overview

This project implements a custom RAG pipeline to understand and demonstrate the core components of Retrieval-Augmented Generation.

The current knowledge source is the research paper:

**Attention Is All You Need**

The system retrieves relevant sections from the document based on the user's query and provides those sections as context to an LLM before generating an answer.

The goal of this Phase-I implementation is to understand the internal RAG workflow instead of treating RAG as a black-box system.

---

## 🧠 What is RAG?

Retrieval-Augmented Generation combines information retrieval with Large Language Models.

Instead of asking the LLM to answer a question using only its internal knowledge, the system first retrieves relevant information from an external knowledge base and provides it as context.

The workflow is:

```text
User Query
    ↓
Query Embedding
    ↓
Vector Similarity Search
    ↓
Relevant Document Chunks
    ↓
RAG Prompt
    ↓
LLM
    ↓
Grounded Answer

This approach can help provide answers based on a specific knowledge source and reduce reliance on the model's general knowledge.

## 🏗️ Architecture


                    Knowledge Source
                          │
                          ▼
                         PDF
                          │
                          ▼
                  Text Extraction
                          │
                          ▼
                       Chunking
                          │
                          ▼
                Sentence Transformer
                     Embeddings
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
        knowledge_base.pkl     FAISS Index
        chunks + embeddings    vector search
                │                   │
                └─────────┬─────────┘
                          │
                          ▼
                     User Query
                          │
                          ▼
                   Query Embedding
                          │
                          ▼
                   FAISS Search
                          │
                          ▼
                    Top-K Chunks
                          │
                          ▼
                     RAG Prompt
                          │
                          ▼
                       LLM API
                          │
                          ▼
                       Answer
```

---

## 🔄 Pipeline

🔄 Pipeline
1. Document Extraction

The PDF text is extracted using pypdf.

The current knowledge source contains 15 pages.

2. Text Chunking

The extracted text is divided into smaller overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

Current configuration:

Chunk size: 500 characters
Chunk overlap: 50 characters
Total chunks: 89

The overlap helps preserve context between neighboring chunks.

3. Embeddings

Each document chunk is converted into a numerical vector using:

all-MiniLM-L6-v2

The resulting embeddings have:

384 dimensions

For the current document:

89 chunks × 384 dimensions
4. Persistent Knowledge Base

The chunks and their embeddings are stored locally in:

data/knowledge_base.pkl

This allows the application to reuse the previously generated chunks and embeddings instead of processing and embedding the PDF every time the application starts.

5. FAISS Vector Search

FAISS is used for semantic similarity search over the document embeddings.

The embeddings are L2-normalized and searched using:

FAISS IndexFlatIP

For normalized vectors, inner product search corresponds to cosine similarity.

The system retrieves the top 3 most relevant chunks for each query.

The FAISS index is persisted locally as:

data/faiss_index.bin

This allows the application to load the existing vector index instead of rebuilding it on every startup.

6. RAG Prompt

The retrieved chunks are combined with the user's question and supplied to the LLM as context.

The prompt instructs the model to answer using only the retrieved context.

7. Answer Generation

The final answer is generated using the OpenAI API.

🛠️ Tech Stack
Python 3.11
pypdf
LangChain Text Splitters
Sentence Transformers
FAISS
NumPy
Scikit-learn
OpenAI API
python-dotenv
Conda
📁 Project Structure
LLM-RAG-Assistant/
│
├── app/
│   ├── extract_text.py
│   ├── chunk_text.py
│   ├── embeddings.py
│   ├── retrieve.py
│   ├── generate_answer.py
│   ├── faiss_test.py
│   └── test_api.py
│
├── documents/
│   └── attention_is_all_you_need.pdf
│
├── data/
│   ├── knowledge_base.pkl
│   └── faiss_index.bin
│
├── notebooks/
│
├── .gitignore
├── requirements.txt
└── README.md

data/ contains generated knowledge-base and FAISS files and is excluded from version control.

⚙️ Setup
1. Clone the repository
git clone https://github.com/ash21agrawal/LLM-RAG-Assistant.git
cd LLM-RAG-Assistant
2. Create the Conda environment
conda create -n llm-rag python=3.11
conda activate llm-rag
3. Install dependencies
python -m pip install -r requirements.txt
4. Configure the OpenAI API key

Create a file:

app/.env

Add:

OPENAI_API_KEY=your_api_key_here

Never commit the .env file to GitHub.

▶️ Running the Project

After cloning the repository, the generated files inside data/ will not be present because they are excluded from version control.

Run:

python app/retrieve.py

On the first run, the system will:

PDF
 ↓
Extract text
 ↓
Create chunks
 ↓
Generate embeddings
 ↓
Save knowledge_base.pkl
 ↓
Create FAISS index
 ↓
Save faiss_index.bin

On subsequent runs, the system loads the existing knowledge base and FAISS index instead of rebuilding them.

The complete RAG application can then be started using:

python app/generate_answer.py

The application accepts interactive questions:

Enter your query (type 'exit' to quit):
💬 Example
Query
What is multi-head attention?
Retrieved Context

The system retrieves the most relevant chunks from the Transformer paper using FAISS similarity search.

Generated Answer
Multi-head attention is a mechanism that runs several attention
layers in parallel. Each head uses learned query, key, and value
projections to compute attention, and the results are combined
to produce the output.

The answer is generated using the retrieved document context.

🔍 Retrieval Configuration
Parameter	Value
Embedding Model	all-MiniLM-L6-v2
Embedding Dimension	384
Number of Chunks	89
Chunk Size	500 characters
Chunk Overlap	50 characters
Vector Index	FAISS IndexFlatIP
Similarity	Cosine similarity via normalized inner product
Top-K Retrieval	3
🎯 Current Capabilities
Extract text from PDF documents
Split documents into overlapping chunks
Generate semantic embeddings
Persist chunks and embeddings
Build and persist a FAISS vector index
Perform semantic similarity search
Retrieve top-K relevant chunks
Construct a context-grounded RAG prompt
Generate answers using an LLM
Interactive terminal-based question answering
🚧 Current Limitations

This is a Phase-I implementation focused on understanding the core RAG pipeline.

Current limitations:

Single knowledge document
Terminal-based interface
No graphical user interface
No user authentication
No multi-user document isolation
No document management interface
No conversational memory
🚀 Future Scope — Phase II

The next phase will transform this core RAG pipeline into a complete user-facing application.

Planned features include:

Web-based frontend
Chatbot interface
Multiple document uploads
PDF / DOCX / TXT support
User-specific document collections
Automatic document ingestion
Persistent vector database
Conversational chat history
Document management
Source-aware responses
Cloud deployment

The goal is to allow users to upload their own documents and ask natural-language questions about the information contained within them.

📚 Knowledge Source

The current demonstration uses the research paper:

Attention Is All You Need

by Ashish Vaswani et al.

The paper is used as the knowledge source for demonstrating the RAG pipeline.

📌 Project Status
Phase I — Core RAG Pipeline

✅ Complete

PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
Persistent Knowledge Base
 ↓
FAISS Vector Search
 ↓
Top-K Retrieval
 ↓
RAG Prompt
 ↓
LLM
 ↓
Answer
Phase II — Multi-Document RAG Application

🚧 Planned

⭐ Key Learning Outcomes

This project was built to understand the internal components of a RAG system rather than relying entirely on a black-box RAG framework.

Key concepts implemented and explored:

Document preprocessing
Text chunking
Chunk overlap
Semantic embeddings
Vector representations
Cosine similarity
FAISS indexing
Top-K retrieval
Persistent knowledge bases
RAG prompt construction
Context-grounded generation
Separation of retrieval and generation layers