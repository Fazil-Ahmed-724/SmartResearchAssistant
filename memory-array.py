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
    temperature=0,
)
# promts = [
#     ("system", "You are a PHP developer."),
#     ("user", "how can i sort the array?")
# ]
# result = llm.invoke(promts)
# print(result.content)

history = []  # Example: [{"role": "user", "content": "Who is pm of india?"}]

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit", "bye"]:
        print("Good Bye 👋")
        break

    history.append({"role": "user", "content": query})
    print("User: ", query)

    res = llm.invoke(history)
    history.append({"role": "ai", "content": res.content})
    print("AI: ", res.content, "\n")
