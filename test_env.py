import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("TOKEN:", token)