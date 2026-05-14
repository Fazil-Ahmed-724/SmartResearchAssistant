import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
    streaming=True,
    temperature=0,
)
prompt = "can you explain about Streaming in langchain-google-genai?"
result = llm.stream(prompt)
for chunk in result:
    print(chunk.content, end="")

