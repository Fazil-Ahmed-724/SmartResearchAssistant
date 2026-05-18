from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, CSVLoader, UnstructuredFileLoader, PyPDFLoader, OnlinePDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WikipediaLoader

load_dotenv()

load = TextLoader("data/knowledge.txt")
texts = load.load()   
oneDoc = texts[0]
print (oneDoc.page_content)

pdf_loader = PyPDFLoader("data/sample.pdf")
pdf_texts = pdf_loader.load()
pdfDoc = pdf_texts[0]
print(pdfDoc.page_content)

csv_loader = CSVLoader("data/products-100.csv")
csv_texts = csv_loader.load()
csvDoc = csv_texts[0]
print(csvDoc.page_content)

web_loader = WebBaseLoader("https://www.w3schools.com/sql/sql_null_values.asp")
web_texts = web_loader.load()
webDoc = web_texts[0]
print(webDoc.page_content)

wik_docs = WikipediaLoader(query="AI", load_max_docs=2).load()
w_doc = wik_docs[0].page_content[:400]  # a part of the page content
print(w_doc)

