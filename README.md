# 🚀 AI-based Document Search & Knowledge Retrieval with Conversational Interface

**Infosys Springboard Virtual Internship Project**  
**RAG (Retrieval-Augmented Generation) Application**

📱 **Live Demo**
**[Try the App](https://document-search-and-knowledge-retrieval.streamlit.app/)**

## 🎯 **Project Overview**
Production-ready **Streamlit + LangChain + Gemini AI** powered RAG app for **semantic document Q&A**.

**Core Technologies**:
- ✅ **Document Loaders**: PDF, TXT, PPTX multi-format support
- ✅ **Lazy Loading**: Memory-optimized streaming  
- ✅ **Vector Embeddings**: Google Gemini Embedding-001
- ✅ **Semantic Search**: Cosine similarity retrieval
- ✅ **Structured Output**: PydanticOutputParser JSON
- ✅ **Source Attribution**: Exact chunk citations
- ✅ **Confidence Scoring**: Model reliability metrics

## ✨ **Features**
| Feature | Details |
|---------|---------|
| **Multi-format Upload** | PDF, TXT, PPTX files |
| **Lazy Document Loading** | Generator-based memory optimization |
| **Top-K Retrieval** | Semantic search from document chunks |
| **Structured Responses** | JSON with answer + sources + confidence |
| **Source Attribution** | Shows exact document chunks retrieved |
| **Confidence Metrics** | Model confidence scores |
| **Chat Memory** | Conversational context preservation |
| **Fast Deployment** | Streamlit Cloud ready |

## 

### **Quick Test Steps**:
1. **Upload any document** (PDF/TXT/PPTX)
2. **Ask any question** about the content
3. **Get structured answer** with sources cited

### **Example Workflows**:
📄 Upload: Research Paper / Article / PPT
🤔 Query: "What are the key findings?"
📊 Output: Answer + Top 3 sources + Confidence 95%


## 🛠️ **Tech Stack**
Frontend: Streamlit (Python)
Backend: LangChain Framework
LLM: Google Gemini 1.5 Flash
Embeddings: Gemini Embedding-001 (3072 dimensions)
Vector Search: Cosine Similarity
Parsing: PydanticOutputParser (Structured Output)
Deployment: Streamlit Cloud



## 🚀 **Local Installation*
```bash
# Clone repository
git clone https://github.com/kogantikarthik/AI-based-Document-Search-and-Knowledge-Retrieval-with-Conversational-Interface.git
cd AI-based-Document-Search-and-Knowledge-Retrieval-with-Conversational-Interface
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows
# Install dependencies
pip install -r requirements.txt
# Setup API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
# Run application
streamlit run final_rag_project.py



📋 Requirements
streamlit>=1.28.0
langchain>=0.1.0
langchain-google-genai>=0.0.5
python-dotenv>=1.0.0
scikit-learn>=1.3.0
pypdf>=3.17.0



📊 How It Works
1. Document Upload
Supports: PDF, TXT, PPTX
Lazy loading for memory efficiency
Automatic text extraction
2. Vector Embedding
Convert text chunks to 3072-dim vectors
Gemini Embedding-001 model
Cosine similarity indexing
3. Query Processing
User question → vector embedding
Semantic search (top-K retrieval)
Chunk ranking by similarity
4. Response Generation
Retrieved chunks → LLM context
Structured output parsing
JSON format: answer + sources + confidence
🎓 Implementation Highlights
LangChain Components:
TextLoader - Load TXT files
PyPDFLoader - Extract PDF content
ChatGoogleGenerativeAI - Gemini LLM
GoogleGenerativeAIEmbeddings - Vector embeddings
PydanticOutputParser - Structured JSON responses




RAG Pipeline:
Document → Chunks → Embeddings → Vector DB → Query → Embedding → Semantic Search → Top-K Chunks → LLM Context → Structured Answer



📚 Project Learning Outcomes
✅ Document loading & text preprocessing
✅ Vector embeddings & semantic search
✅ LLM integration & prompt engineering
✅ Structured output parsing (Pydantic)
✅ Production-ready RAG systems
✅ Cloud deployment (Streamlit)



Author: Koganti Karthik Chowdary
College: 3rd Year BTech - Computer Science( Artificial Intelligence And Machine Learning)
Internship: Infosys Springboard 2026
Contact: kogantikarthik21@gmail.com
