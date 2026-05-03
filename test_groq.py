import os
from groq import Groq

# Initialize the client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain why Groq is so fast.",
        }
    ],
    model="gpt-oss-20b",
)

print(chat_completion.choices[0].message.content)