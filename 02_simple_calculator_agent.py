"""
Module 2: Simple Tool-Using Agent
Demonstrates rule-based tool routing:
- If user query requests a calculation, route to custom calculator function.
- Otherwise, route to LLM for general knowledge response.
"""

import os
import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[Error] GROQ_API_KEY is not set. Please update your .env file.")
    sys.exit(1)

client = Groq(api_key=api_key)

def ask_llm(prompt: str) -> str:
    """Fallback LLM query function."""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful AI teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def calculator(expression: str) -> str:
    """Safely evaluates basic mathematical expressions."""
    # Whitelist safe characters for arithmetic
    allowed = set("0123456789+-*/(). %")
    if not all(c in allowed for c in expression):
        return "Error: Expression contains unsupported characters."
    try:
        # Safe eval with restricted globals/locals
        result = eval(expression, {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def simple_agent(user_input: str) -> str:
    """Agent router deciding between tool execution and LLM response."""
    if "calculate" in user_input.lower():
        expression = user_input.lower().replace("calculate", "").strip()
        result = calculator(expression)
        return f"The calculated result is: {result}"
    else:
        return ask_llm(user_input)

if __name__ == "__main__":
    queries = [
        "calculate 25 * 4 + 10",
        "What is an AI agent explain in 50 words?"
    ]
    for q in queries:
        print("=" * 60)
        print(f"Query: {q}")
        print("=" * 60)
        print(simple_agent(q))
        print()
