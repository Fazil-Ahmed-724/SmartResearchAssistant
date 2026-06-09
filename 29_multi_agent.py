import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langchain_core.tools import tool



from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

load_dotenv()  # Load environment variables from .env file

class FlowState(BaseModel):
    question:str = Field(description="The user's question")
    category: Literal['coding', 'google_search', 'weather'] = Field(default='coding', description="The category of the question")
    answer:str = Field(default="", description="The answer to the question")

class QuestionCategory(BaseModel):
    category: Literal['coding', 'google_search', 'weather'] = Field(default='coding', description="The category of the question")

# Fetch API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "API key not found. Set GROQ_API_KEY in your environment or .env file."
    )

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
)

def check_question_category(state: FlowState) -> FlowState:
    st_llm = llm.with_structured_output(QuestionCategory)
    res = st_llm.invoke(f"Categorize this question into one of: coding, google_search, or weather. Question: {state.question}")

    print(f"Category detected: {res.category}")
    state.category = res.category
    return state

def route(state: FlowState) -> Literal['coding', 'google_search', 'weather']:
    # This function is not strictly necessary since we can directly use the category in the graph edges, but it can be useful for clarity or additional processing if needed.
    return state.category

def coding_node(state: FlowState) -> FlowState:
    response = llm.invoke(f"your a coding expert {state.question}")
    state.answer = response.content
    return state

def google_search_node(state: FlowState) -> FlowState:
    response = google_agent.invoke({"messages": [{"role": "user", "content": state.question}]})
    state.answer = response["messages"][-1].content
    return state

def weather_node(state: FlowState) -> FlowState:
    state.answer = weather_agent.invoke({"messages": [{"role": "user", "content": state.question}]})["messages"][-1].content
    return state
# Define tools for agents
@tool
def google_search(query: str) -> str:
    """A tool to perform a Google search for a given query."""
    return search.run(query)
# Initialize agents
search = GoogleSerperAPIWrapper()
google_agent = create_agent(
    model=llm,
    tools=[google_search],
    system_prompt="You are a helpful assistant that can perform Google searches to answer user queries.",
)
# Define a tool for getting weather information
@tool
def get_weather(location: str) -> str:
    """A tool to get the current weather for a given location."""
    return f"The current weather in {location} is sunny with a temperature of 25°C."  # Placeholder response
# In a real implementation, you would integrate with a weather API to get actual data.
weather_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can provide weather information.",
)
# Define the state graph
graph = StateGraph(state_schema=FlowState)
graph.add_node("categorize", check_question_category)
graph.add_node("coding", coding_node)
graph.add_node("google_search", google_search_node)
graph.add_node("weather", weather_node)
# Define edges based on the category
graph.add_edge(START, "categorize")
graph.add_conditional_edges("categorize", route, {
    'coding': 'coding',
    'google_search': 'google_search',
    'weather': 'weather'
})
graph.add_edge("coding", END)
graph.add_edge("google_search", END)
graph.add_edge("weather", END)

# Compile the graph
graph = graph.compile()
# Test the graph
# initial_state = FlowState(question="what is the weather in new york?")
result = graph.invoke({'question': "help me to find car in karachi"})
print("Final Result:", result)
print("\n--- Answer ---")
print(result['answer'])
