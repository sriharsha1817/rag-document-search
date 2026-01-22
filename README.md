# 🚀 AI-based Document Search & Knowledge Retrieval with Conversational Interface

**Infosys Springboard Virtual Internship Project**  
**RAG (Retrieval-Augmented Generation) Application**

Live Demo:
https://rag-document-search-bgc2rwcbs9ewt4yamckfbh.streamlit.app/

This repository contains my Retrieval-Augmented Generation (RAG) project that enables users to upload different types of documents and have meaningful conversations about their contents using semantic search and large language models (LLMs).

🚀 Project Overview

This is an AI-powered document understanding application built with Python, Streamlit, LangChain, and Google Gemini.
The application allows users to upload documents (PDF, TXT, PPTX) and ask questions about their contents, returning structured answers with source attribution and confidence scores.

This project was developed as part of the Infosys Springboard Virtual Internship 2026.

🧠 Core Functionality

Upload and process files in PDF, TXT, and PPTX formats

Perform semantic search using vector embeddings

Retrieve the most relevant document chunks in real time

Generate context-aware answers using an LLM

Provide source citations and confidence metrics

Preserve conversational context across multiple queries

🛠️ Tech Stack

Frontend

Streamlit (Python)

Backend & AI

LangChain

Google Gemini 1.5 Flash (LLM)

Vector Search

Google Gemini Embedding-001 (3072 dimensions)

Cosine similarity semantic search

Structured Output

PydanticOutputParser for JSON responses

Deployment

Streamlit Cloud

🔍 How It Works

A user uploads a document (PDF, TXT, PPTX)

The document is loaded and chunked

Each chunk is embedded into a vector

Vectors are stored in a vector database

User queries are embedded and matched

Top-K relevant chunks are retrieved

The LLM generates a structured answer

Answer + sources + confidence score are shown

📁 Supported Document Formats

PDF files

Plain text (TXT)

PowerPoint (PPTX)

🧪 Features

✔ Multi-format upload support
✔ Memory-efficient lazy document processing
✔ Top-K semantic retrieval
✔ Structured answers with JSON formatting
✔ Source citation for every answer
✔ Confidence scoring
✔ Conversational memory

📌 Example Workflow

Upload a research paper, article, or presentation

Ask:
“What are the key findings of this document?”

Get a response like:

Answer summary

Top-3 relevant chunks

Confidence score

🧾 Dependencies
streamlit>=1.28.0
langchain>=0.1.0
langchain-google-genai>=0.0.5
python-dotenv>=1.0.0
scikit-learn>=1.3.0
pypdf>=3.17.0

🖥️ Running the Project Locally
Step 1 — Clone the Repo
git clone https://github.com/sriharsha1817/rag-document-search.git
cd rag-document-search

Step 2 — Set Up Virtual Environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Step 3 — Install Dependencies
pip install -r requirements.txt

Step 4 — Add Environment Variables
cp .env.example .env


Open .env and add your Google API key (for Gemini)

Step 5 — Run the App
streamlit run final_rag_project.py


Visit the local URL displayed in your terminal to interact with the app.

🛠️ Key Implementation Components

Document Loaders

TextLoader – Reads TXT files

PyPDFLoader – Extracts text from PDFs

Embeddings & Retrieval

GoogleGenerativeAIEmbeddings – Embeddings

Cosine similarity for semantic search

Conversational AI

ChatGoogleGenerativeAI – Generates answers

PydanticOutputParser – Structured JSON

🧠 What I Learned

While building this project I gained hands-on experience with:

Document parsing and preprocessing

Vector embeddings and semantic search

LLM integration and prompt engineering

Structured output parsing using Pydantic

Building production-ready RAG systems

Deploying AI applications on cloud platforms

📬 Contact

Developed by: Bommineni Sri Harsha
Email: sriharsha8171@gmail.com
]
