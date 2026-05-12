import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variables
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file."
    )

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key  # Fetch API key from environment variable
)

# App Title
st.title("🤖 AskBuddy - AI QnA Bot")
st.markdown("My QnA Bot with LangChain and Google Gemini!")

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

# Chat input
query = st.chat_input("Ask anything ?")

# Handle user input
if query:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Get AI response
    response = llm.invoke(query)

    ai_response = response.content

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Save AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })