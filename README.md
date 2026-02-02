
# Local LLM with PDF RAG (Streamlit + Ollama + Chroma)

This project is a **local PDF-based question answering app**. It allows you to:

- Upload PDFs
- Index them into a vectorstore using embeddings
- Ask questions about the PDF
- Get answers from a local LLM (`llama3.2`) without sending data to the cloud

All processing happens **locally on your machine**.

---

## Features

- **PDF Upload & RAG**: Reads and indexes PDF documents into chunks
- **Vectorstore**: Uses Chroma to store embeddings
- **Local LLM**: Queries PDF content using `llama3.2` via Ollama
- **Streaming Chat**: Shows the answer in real-time while the LLM is generating it
- **Chat History**: Maintains conversation context

---

## Requirements

- Python 3.12
- Ollama installed and running
- macOS (tested) / Linux compatible

### Python Dependencies

```bash
pip install --upgrade pip
pip install streamlit
pip install langchain>=0.1.0
pip install langchain-community
pip install chromadb
pip install ollama
pip install PyPDF2
