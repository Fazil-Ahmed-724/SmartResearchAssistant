import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langchain.agents import create_agent
from typing import Annotated

from regex import search
load_dotenv()  # Load environment variables from .env file  
class chatState(BaseModel):
    message: Annotated [list, add_messages]
       
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
# Initialize memory saver
memory = InMemorySaver()

# Node 1: Start
def chatBotNode(state: chatState) -> chatState:
    res = llm.invoke(state.message)
    state.message = [res]
    return state    

# create graph
graph = StateGraph(chatState)
# add nodes and edges
graph.add_node('chatBot', chatBotNode)

graph.add_edge(START, 'chatBot')
graph.add_edge('chatBot', END)

# compile graph
final_graph = graph.compile(checkpointer=memory)

# Test the graph
config = {"configurable": {"thread_id": "1"}}    
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break
    result = final_graph.invoke({"message": [{"role": "user", "content": user_input}]}, config=config)
    print(result["message"][-1].content)
    


