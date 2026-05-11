import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
    temperature=0,
)

# Dynamic Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {language} developer."),
    ("user", "{question}")
])

# Create chain
chain = prompt | llm

# Dynamic input
language = input("Enter language: ")
question = input("Enter question: ")

# Invoke chain
result = chain.invoke({
    "language": language,
    "question": question
})

print(result.content)
