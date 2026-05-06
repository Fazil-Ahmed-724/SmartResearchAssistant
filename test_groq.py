import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or export it in your shell.")

client = Groq(
    api_key=api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What's the weather in Karachi on 2023-10-01?",
        }
    ],
    model="openai/gpt-oss-20b",
)

print(chat_completion.choices[0].message.content)
