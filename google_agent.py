import os
from typing import Any

from dotenv import load_dotenv
from google.genai import Client
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return api_key


def maybe_list_models(api_key: str, limit: int = 10) -> None:
    """Optionally print available Google models when LIST_GOOGLE_MODELS=true."""
    if os.getenv("LIST_GOOGLE_MODELS", "").lower() != "true":
        return

    try:
        client = Client(api_key=api_key)
        print("Available models:")
        for index, model in enumerate(client.models.list(), start=1):
            description = getattr(model, "description", "") or "No description available."
            print(f"- {model.name}: {description}")
            if index >= limit:
                break
    except Exception as exc:
        print(f"Could not list Google models: {exc}")


def build_agent():
    api_key = get_google_api_key()
    model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash-lite")

    maybe_list_models(api_key)

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
    )

    return create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt=(
            "You are a helpful assistant. Use the weather tool when a user asks "
            "about current weather."
        ),
    )


def extract_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text:
        return text

    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return str(content)


def main() -> None:
    agent = build_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's the weather in San Francisco?",
                }
            ]
        }
    )
    print(extract_text(result["messages"][-1]))


if __name__ == "__main__":
    main()
