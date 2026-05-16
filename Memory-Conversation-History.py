import os
from urllib import response
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
    

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variables
api_key = os.getenv("SERPER_API_KEY")
if not api_key:
    raise RuntimeError(
        "API key not found. Set SERPER_API_KEY in your environment or .env file."
    )

# Fetch API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "API key not found. Set GROQ_API_KEY in your environment or .env file."
    )

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",
)

search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm,
    tools=[search.run],
    system_prompt="You are a helpful assistant that can perform Google searches to answer user queries.",
    checkpointer=InMemorySaver(),

)
while True:
    question = input("Enter your question (or 'exit' to quit): ")
    if question.lower() == "exit":
        break

    response = agent.invoke(
        {
            "messages": [
                {"role": "user", "content": question},
            ]
        },
        {"configurable": {"thread_id": "1"}},

    )
    print(response["messages"][-1].content)
