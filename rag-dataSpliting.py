from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader, PyPDFLoader, OnlinePDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

load = TextLoader("data/knowledge.txt")
texts = load.load() 
doc_splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = doc_splitters.split_documents([texts[0]])
print (docs[0].page_content)

pdf_loader = PyPDFLoader("data/sample.pdf")
pdf_texts = pdf_loader.load()
pdf_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pdf_docs = pdf_spliter.split_documents([pdf_texts[0]])
print(pdf_docs[0])

csv_loader = CSVLoader("data/products-100.csv")
csv_texts = csv_loader.load()
csv_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
csv_docs = csv_spliter.split_documents([csv_texts[0]])
print(csv_docs[0])

web_loader = WebBaseLoader("https://www.w3schools.com/sql/sql_null_values.asp")
web_texts = web_loader.load()
web_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
web_docs = web_spliter.split_documents([web_texts[0]])
print(web_docs[0])

wik_docs = WikipediaLoader(query="AI", load_max_docs=2).load()
wik_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
wik_docs_split = wik_spliter.split_documents([wik_docs[0]])
print(wik_docs_split[0])

