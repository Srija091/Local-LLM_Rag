# rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings


def load_pdf(pdf_path):
    """
    Load a PDF, split it into chunks, embed it with Ollama, 
    and store it in a Chroma vectorstore.
    """
    # 1️⃣ Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2️⃣ Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3️⃣ Generate embeddings
    embeddings = OllamaEmbeddings(model="llama3.2")  # make sure this model is installed locally

    # 4️⃣ Create Chroma vectorstore
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings
    )

    return vectorstore
