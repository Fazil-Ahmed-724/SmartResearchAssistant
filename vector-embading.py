from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv() 


class SafeGoogleGenerativeAIEmbeddings(Embeddings):
    """Force single-item batching so document counts match embedding counts."""

    def __init__(self, base_embeddings: GoogleGenerativeAIEmbeddings):
        self.base_embeddings = base_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base_embeddings.embed_documents(texts, batch_size=1)

    def embed_query(self, text: str) -> list[float]:
        return self.base_embeddings.embed_query(text)


documents = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Python is a programming language popular for data science",
        "India won the cricket world cup in 2011",
        "AI models can understand and generate human language"
    ]

base_embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
embeddings_model = SafeGoogleGenerativeAIEmbeddings(base_embeddings_model)


vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embeddings_model,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

query = "What is machine learning?"
results = vector_store.similarity_search(query)
print("Query:", query)
print("Results:")
for result in results:
    print("-", result.page_content)
