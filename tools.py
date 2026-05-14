import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )


@tool
def add_number(a: int, b: int):
    """Adds two numbers and returns the result."""
    return a + b


@tool
def multiply_number(a: int, b: int):
    """Multiplies two numbers and returns the result."""
    return a * b


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=api_key,
    temperature=0,
)

agent = create_agent(
    model=llm,
    tools=[add_number, multiply_number],
    system_prompt="You are a helpful assistant that can perform basic arithmetic operations.",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What is 2 + 3?"},
            {"role": "user", "content": "What is 4 * 5?"},
        ]
    }
)

final_answer = None
for message in reversed(result["messages"]):
    if isinstance(message, AIMessage) and message.content:
        final_answer = message.content
        break
    if isinstance(message, ToolMessage) and final_answer is None:
        final_answer = message.content

print(final_answer)
