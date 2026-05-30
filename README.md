---
title: PDF Chat RAG
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.36.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
---


# 📚 RAG-based Document Q&A System

> An AI-powered chatbot that lets you upload any PDF and ask questions about its content. Built with LangChain, ChromaDB, and Llama 3.3 70B.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green)
![Gradio](https://img.shields.io/badge/Gradio-4.36-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Live-success)

## 🚀 Live Demo

**Try it now:** [https://huggingface.co/spaces/Abubakr4t/pdf-chat-rag](https://huggingface.co/spaces/Abubakr4t/pdf-chat-rag)

No setup required — upload any PDF and start asking questions.

## 🎯 What It Does

Upload a PDF document (research paper, textbook chapter, manual, report) and ask questions in natural language. The system:

1. Parses and splits your PDF into semantic chunks
2. Converts each chunk into vector embeddings
3. Stores them in a vector database (ChromaDB)
4. Retrieves the most relevant chunks for any question
5. Generates accurate, grounded answers using Llama 3.3 70B
6. Cites source pages so you can verify every claim

## ✨ Features

- 📤 **Upload any PDF** — research papers, books, manuals, reports
- 💬 **Natural language Q&A** — ask questions like you would to a human
- 📖 **Source citations** — every answer shows which pages it came from
- 🚫 **No hallucinations** — refuses to answer if info isn't in the document
- 🎨 **Polished UI** — clean, responsive Gradio interface
- ⚡ **Fast inference** — Llama 3.3 70B via Groq API (lowest latency LLM provider)
- 🆓 **Free to use** — no API keys required for end users

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Llama 3.3 70B (via Groq API) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Vector Store** | ChromaDB |
| **Framework** | LangChain |
| **UI** | Gradio 4.36 |
| **Deployment** | Hugging Face Spaces |
| **Language** | Python 3.11 |

## 📐 Architecture


┌─────────────────┐
│  User uploads   │
│      PDF        │
└────────┬────────┘
│
▼
┌─────────────────┐      ┌─────────────────────┐
│  PyPDFLoader    │ ───▶ │  RecursiveCharacter │
│  parses PDF     │      │  TextSplitter       │
└─────────────────┘      │  (1000 chars,       │
│   200 overlap)      │
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│  HuggingFace        │
│  Embeddings         │
│  (MiniLM-L6-v2)     │
│  → 384-dim vectors  │
└──────────┬──────────┘
│
▼
┌─────────────────────┐
│  ChromaDB           │
│  Vector Store       │
└──────────┬──────────┘
│
┌─────────────────┐                 │
│  User asks      │                 │
│  question       │                 │
└────────┬────────┘                 │
│                          │
▼                          │
┌─────────────────┐                 │
│  Semantic       │  ◀──────────────┘
│  Retrieval      │
│  (top-k=6)      │
└────────┬────────┘
│
▼
┌─────────────────┐      ┌─────────────────────┐
│  Llama 3.3 70B  │ ───▶ │  Final Answer       │
│  via Groq API   │      │  with citations     │
└─────────────────┘      └─────────────────────┘




## 🔧 How It Works (Technical Detail)

1. **Document Ingestion:** PDF is parsed using `PyPDFLoader`, extracting text with page metadata
2. **Chunking:** Text is split into 1000-character chunks with 200-character overlap using `RecursiveCharacterTextSplitter`. Overlap prevents loss of meaning at chunk boundaries.
3. **Embedding:** Each chunk is converted to a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`
4. **Storage:** Vectors stored in `ChromaDB` with unique collection per document
5. **Retrieval:** User question is embedded and matched against stored vectors using cosine similarity. Top 6 chunks retrieved.
6. **Generation:** Retrieved chunks injected into a custom prompt template and sent to Llama 3.3 70B for answer generation
7. **Citation:** Source chunks displayed with page numbers for verification

## 🏃 Running Locally

```bash
# Clone the repo
git clone https://github.com/abubakr4t/rag-document-qa.git
cd rag-document-qa

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your_groq_api_key_here"

# Run the app
python app.py
```

Then open `http://localhost:7860` in your browser.

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## 🔮 Future Improvements

- [ ] **Hybrid search** — combine semantic (dense) + keyword (BM25) retrieval
- [ ] **Cross-encoder reranking** — improve retrieval precision with reranking model
- [ ] **Conversation memory** — support follow-up questions in context
- [ ] **Multi-PDF support** — query across multiple documents simultaneously
- [ ] **RAGAS evaluation** — measure faithfulness, answer relevancy, context precision
- [ ] **Streaming responses** — token-by-token streaming for better UX
- [ ] **OCR support** — handle scanned/image-based PDFs

## 📊 Performance Notes

- **Latency:** ~2-4 seconds per query (network-bound, Groq is fast)
- **Cost:** Free tier of Groq supports the demo
- **Capacity:** Tested with PDFs up to 100 pages successfully

## 👨‍💻 Developers

- **Muhammad Abubakar** — [LinkedIn](https://www.linkedin.com/in/abubakr4t)
- **Muhammad Zeeshan Asif** — [LinkedIn](https://www.linkedin.com/in/zeeshan-asif-75bb57312)

## 📄 License

MIT License — feel free to use this code in your own projects.

## 🙏 Acknowledgments

- Groq for blazing-fast LLM inference
- Meta AI for the Llama 3.3 model
- LangChain team for the RAG framework
- Hugging Face for free hosting
