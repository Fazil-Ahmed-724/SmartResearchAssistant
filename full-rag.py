from pathlib import Path
import re
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

UPLOAD_DIR = Path("data-upload")
VECTOR_DB_DIR = "./chroma_langchain_db"
STOPWORDS = {
    "about",
    "agent",
    "answer",
    "document",
    "from",
    "full",
    "give",
    "info",
    "information",
    "is",
    "me",
    "of",
    "pdf",
    "tell",
    "the",
    "this",
    "what",
}

if "document_upload" not in st.session_state:
    st.session_state.document_upload = False

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "chat_model" not in st.session_state:
    st.session_state.chat_model = None


class SafeGoogleGenerativeAIEmbeddings(Embeddings):
    """Force single-item batching so document counts match embedding counts."""

    def __init__(self, base_embeddings: GoogleGenerativeAIEmbeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embeddings.embed_documents(texts, batch_size=1)

    def embed_query(self, text: str) -> list[float]:
        return self.base_embeddings.embed_query(text)


def load_pdf_documents(file_paths: list[Path]) -> list[Document]:
    documents: list[Document] = []
    for file_path in file_paths:
        documents.extend(PyPDFLoader(str(file_path)).load())
    return documents


def build_context(query: str, limit: int = 8) -> list[Document]:
    vector_store = st.session_state.vector_store
    chunks: list[Document] = st.session_state.document_chunks

    if vector_store is None or not chunks:
        return []

    max_results = min(limit, len(chunks))
    results = vector_store.similarity_search(query, k=max_results)

    combined: list[Document] = []
    seen: set[tuple[str, int, int]] = set()

    def add_doc(doc: Document) -> None:
        source = doc.metadata.get("source", "")
        page = int(doc.metadata.get("page", -1))
        start_index = int(doc.metadata.get("start_index", -1))
        key = (source, page, start_index)
        if key not in seen:
            seen.add(key)
            combined.append(doc)

    for doc in results:
        add_doc(doc)

    query_terms = [
        term.lower()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", query)
        if term.lower() not in STOPWORDS
    ]

    for doc in chunks:
        content = doc.page_content.lower()
        if any(term in content for term in query_terms):
            add_doc(doc)
        if len(combined) >= 12:
            break

    return combined[:12]


def answer_question(query: str) -> str:
    context_docs = build_context(query)
    if not context_docs:
        return "I could not find any indexed PDF content yet. Please upload a PDF first."

    context_blocks = []
    for index, doc in enumerate(context_docs, start=1):
        page_label = doc.metadata.get("page_label") or str(int(doc.metadata.get("page", 0)) + 1)
        source_name = Path(doc.metadata.get("source", "uploaded.pdf")).name
        context_blocks.append(
            f"[Source {index} | {source_name} | page {page_label}]\n{doc.page_content.strip()}"
        )

    prompt = f"""
You are a helpful assistant answering questions strictly from the uploaded PDF context.

Rules:
- Answer only from the provided context.
- If the user asks about a person, company, skill, or project, combine all relevant mentions across the context.
- If the answer is not present, say you could not find it in the uploaded PDF.
- Include page references when possible.

Question:
{query}

Context:
{chr(10).join(context_blocks)}
""".strip()

    response = st.session_state.chat_model.invoke(prompt)
    if isinstance(response.content, str):
        return response.content
    return str(response.content)


def process_documents(file_paths: list[Path]) -> None:
    pdf_texts = load_pdf_documents(file_paths)

    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    pdf_docs = pdf_splitter.split_documents(pdf_texts)

    base_embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    embeddings_model = SafeGoogleGenerativeAIEmbeddings(base_embeddings_model)

    vector_db = Chroma.from_documents(
        documents=pdf_docs,
        embedding=embeddings_model,
        persist_directory=VECTOR_DB_DIR,
        collection_name=f"upload_{uuid4().hex}",
    )

    st.session_state.vector_store = vector_db
    st.session_state.document_chunks = pdf_docs
    st.session_state.indexed_files = [file_path.name for file_path in file_paths]
    st.session_state.chat_model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    st.session_state.document_upload = True
    st.session_state.messages = []


st.title("PDF Q&A")

if not st.session_state.document_upload:
    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        with st.spinner("Processing document..."):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            saved_files: list[Path] = []
            for uploaded_file in uploaded_files:
                target_path = UPLOAD_DIR / uploaded_file.name
                with open(target_path, "wb") as file_handle:
                    file_handle.write(uploaded_file.getvalue())
                saved_files.append(target_path)
            process_documents(saved_files)
            st.rerun()

if st.session_state.document_upload and st.session_state.vector_store and st.session_state.chat_model:
    st.caption(f"Indexed files: {', '.join(st.session_state.indexed_files)}")

    if st.button("Upload different PDF(s)"):
        st.session_state.document_upload = False
        st.session_state.vector_store = None
        st.session_state.document_chunks = []
        st.session_state.indexed_files = []
        st.session_state.chat_model = None
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    query = st.chat_input("Ask a question about the document")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        try:
            answer = answer_question(query)
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as error:
            st.error(f"Error answering question: {error}")
