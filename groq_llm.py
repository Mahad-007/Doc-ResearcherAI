# groq_llm.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # or any other supported model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
