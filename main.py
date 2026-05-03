import os

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


def build_model():
    repo_id = os.getenv("HUGGINGFACE_REPO_ID", "mistralai/Mistral-7B-Instruct-v0.2")
    provider = os.getenv("HUGGINGFACE_PROVIDER", "auto")
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        provider=provider,
        temperature=0.5,
        max_new_tokens=256,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=llm)


def should_search(query):
    prompt = query.lower()
    keywords = (
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "price",
        "who is",
        "what is",
        "when did",
        "where is",
    )
    return any(keyword in prompt for keyword in keywords)


def build_messages(query, search_context):
    system_text = (
        "You are a helpful research assistant. Answer clearly and concisely. "
        "If search results are provided, use them as context and say when the "
        "answer is based on those results."
    )

    if search_context:
        user_text = (
            f"Question: {query}\n\n"
            f"Search results:\n{search_context}\n\n"
            "Use the search results when relevant and provide a helpful answer."
        )
    else:
        user_text = query

    return [
        ("system", system_text),
        ("human", user_text),
    ]


def main():
    print("Smart Research Assistant is running... (type exit to stop)")

    try:
        model = build_model()
        search = DuckDuckGoSearchRun()
    except Exception as exc:
        print("\nStartup error:", exc)
        raise SystemExit(1)

    while True:
        query = input("\nAsk: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:
            search_context = ""
            if should_search(query):
                search_context = search.run(query)

            response = model.invoke(build_messages(query, search_context))
            print("\nAnswer:\n", response.content)
        except Exception as exc:
            if "404 Not Found" in str(exc) or "model_not_supported" in str(exc):
                print(
                    "\nRequest failed: the selected Hugging Face model/provider "
                    "combination is unavailable. Try setting HUGGINGFACE_REPO_ID "
                    "and HUGGINGFACE_PROVIDER in .env to a supported pair."
                )
                continue
            print("\nRequest failed:", exc)


if __name__ == "__main__":
    main()
