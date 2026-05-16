import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to your .env file.")
    return value


def extract_message_text(message: Any) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts).strip()

    return str(content)


@st.cache_resource
def build_agent():
    serper_api_key = get_required_env("SERPER_API_KEY")
    groq_api_key = get_required_env("GROQ_API_KEY")
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information."""
        return search.run(query)

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="openai/gpt-oss-20b",
    )

    return create_agent(
        model=llm,
        tools=[web_search],
        checkpointer=InMemorySaver(),
        system_prompt=(
            "You are a helpful assistant that can search the web to answer "
            "questions. Use the search tool when current or factual information "
            "would help."
        ),
    )


def main() -> None:
    st.set_page_config(page_title="Smart Research Assistant")
    st.title("Smart Research Assistant")
    st.write("Ask anything. The assistant can search the web when needed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = "streamlit-chat-1"

    try:
        agent = build_agent()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask anything")

    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": query},
                    ]
                },
                {"configurable": {"thread_id": st.session_state.thread_id}},
            )
            ai_response = extract_message_text(response["messages"][-1])
            st.markdown(ai_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": ai_response}
    )


main()
