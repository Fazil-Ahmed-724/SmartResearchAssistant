import os
from urllib import response
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import sql_database
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st 

load_dotenv()

db = sql_database.SQLDatabase.from_uri("sqlite:///my_tasks.db")
db.run("""
        CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT CHECK (status IN ('pending', 'in_progress', 'completed')) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
      """);
# Fetch API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError(
        "API key not found. Set GROQ_API_KEY in your environment or .env file."
    )

model = chat_groq = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="openai/gpt-oss-20b",

)
toolkit = SQLDatabaseToolkit(db=db, llm=ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b"))
tools = toolkit.get_tools()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table.

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clear and organized display in the browser.

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource
def get_agent():
      agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
      )
      return agent

agent = get_agent()

st.subheader("SQL Task Management Agent")

if "messages" not in st.session_state:
      st.session_state.messages = []

for message in st.session_state.messages:
      st.chat_message(message["role"]).markdown(message["content"])      


prompt = st.chat_input("Ask me to manage your tasks! (e.g., 'Add a new task to buy groceries', 'Show me my pending tasks', 'Mark task 1 as completed', 'Delete task 2')")

if prompt:
      st.chat_message("user").write(prompt)
      st.session_state.messages.append({"role": "user", "content": prompt})
      with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                  result = agent.invoke(
                        {
                              "messages": [
                              {"role": "user", "content": prompt},
                              ]
                        },
                        {"configurable": {"thread_id": "1"}},
                  )
                  st.markdown(result["messages"][-1].content)
                  st.session_state.messages.append({"role": "assistant", "content": result["messages"][-1].content})
