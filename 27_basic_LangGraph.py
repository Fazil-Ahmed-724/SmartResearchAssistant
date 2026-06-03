from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class MyState(BaseModel):
    message: str = ""

graph = StateGraph(MyState)

# Node 1: Start
def start_node(state: MyState):
    state.message = f"Hello, World! (Input: {state.message})"
    print(f"start_node: {state.message}")
    return state

def end_node(state: MyState):
    state.message = "Goodbye, World!"
    print(f"end_node: {state.message}")
    return state

graph.add_node('start', start_node)
graph.add_node('end', end_node)


graph.add_edge(START, 'start')
graph.add_edge('start', 'end')
graph.add_edge('end', END)

final_graph = graph.compile()
# # Test the graph
result = final_graph.invoke({"message": "hi there"})
# print(result)  # Output: Hello, World!
print(result)  # Output: Hello, World! (Input: hi there)
