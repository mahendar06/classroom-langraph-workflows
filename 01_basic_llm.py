"""
Module 1: Basic LLM Integration using Groq
Demonstrates direct API connection to Groq's high-speed inference engine
using the LLaMA 3.1 model.
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[Error] GROQ_API_KEY is not set in environment or .env file.")
    print("Please create a .env file based on .env.example with your Groq API key.")
    sys.exit(1)

client = Groq(api_key=api_key)

def ask_llm(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.3) -> str:
    """Send a prompt to Groq LLM and return the assistant response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = "Explain AI Agent in 100 words to a 1st grade student."
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)
    result = ask_llm(prompt)
    print(result)
