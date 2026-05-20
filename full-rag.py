import streamlit as st
from dotenv import load_dotenv
from langchain_core import documents
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader, PyPDFLoader, OnlinePDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

if "document_upload" not in st.session_state:
    st.session_state.document_upload = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:    
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


def process_document(path):
    # load documents
    pdf_loader = PyPDFDirectoryLoader(path)
    pdf_texts = pdf_loader.load()

    # split documents
    pdf_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = pdf_spliter.split_documents(pdf_texts)
    # create embeddings
    class SafeGoogleGenerativeAIEmbeddings(Embeddings):
        """Force single-item batching so document counts match embedding counts."""

        def __init__(self, base_embeddings: GoogleGenerativeAIEmbeddings):
            self.base_embeddings = base_embeddings

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.base_embeddings.embed_documents(texts, batch_size=1)

        def embed_query(self, text: str) -> list[float]:
            return self.base_embeddings.embed_query(text)

    base_embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    embeddings_model = SafeGoogleGenerativeAIEmbeddings(base_embeddings_model)

    # vector DB
    vectro_db = Chroma.from_documents(
        documents=pdf_docs,
        embedding=embeddings_model,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )



    # retrieve relevant documents from vector DB
    @tool
    def retrieve_context(query: str):
        """Retrieve relevant documents from the vector database based on the query."""
        context = ""
        results = vectro_db.similarity_search(query, k=4)
        for result in results:
            context = result.page_content + "\n" + context
        return context

    # create agent
    memory = InMemorySaver()
    agent = create_agent(
        model=ChatGroq(model="openai/gpt-oss-20b", temperature=0),
        tools=[retrieve_context],
        system_prompt=(
            "You are a helpful assistant. Use the vector database to answer questions about the PDF document."
        ),
        checkpointer=memory,
    )
    st.session_state.agent = agent
    st.session_state.document_upload = True

# upload UI
if not st.session_state.document_upload:
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=True)
    if uploaded_file:
        with st.spinner("Processing document..."): 
            path = f"data-upload/"
            for file in uploaded_file:
                with open(path+file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()
if st.session_state.document_upload and st.session_state.agent:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)

    query = st.text_input("Ask a question about the document:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        # Use the 'invoke' method to interact with the agent
        try:
            result = st.session_state.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query,
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "streamlit-chat-1",  # Example thread ID
                        "checkpoint_ns": "default_namespace",  # Example namespace
                        "checkpoint_id": "default_checkpoint"  # Example checkpoint ID
                    }
                }
            )
            answer = result["messages"][-1].content
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error invoking agent: {e}")