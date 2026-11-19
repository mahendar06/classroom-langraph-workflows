"""
Module 3: Web Research Agent
Demonstrates integrating DuckDuckGo web search with Groq LLM
to perform structured startup market analysis.
"""

import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
from groq import Groq

# Support both duckduckgo_search and ddgs package naming
try:
    from duckduckgo_search import DDGS
except ImportError:
    try:
        from ddgs import DDGS
    except ImportError:
        DDGS = None

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("[Error] GROQ_API_KEY is not set. Please update your .env file.")
    sys.exit(1)

client = Groq(api_key=api_key)

def ask_llm(prompt: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.3) -> str:
    """Send a prompt to Groq LLM and return the assistant response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a startup research analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Query DuckDuckGo and return titles, URLs, and snippet bodies."""
    if DDGS is None:
        return [{"title": "Search Unavailable", "href": "", "body": "duckduckgo_search is not installed."}]
    
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")
                })
    except Exception as e:
        print(f"[Warning] Web search encountered an issue: {e}")
        results.append({
            "title": "Fallback",
            "href": "",
            "body": f"Search could not fetch external results due to network/rate limit: {e}"
        })
    return results

def startup_research_agent(idea: str) -> str:
    """Agent that performs live web research and synthesizes market insights."""
    search_data = web_search(f"{idea} market trends competitors", max_results=5)

    context = "\n\n".join([
        f"Title: {item['title']}\nSummary: {item['body']}\nURL: {item['href']}"
        for item in search_data
    ])

    prompt = f"""
You are a startup research analyst.

Startup idea:
{idea}

Web research data:
{context}

Provide a structured assessment covering:
1. Market opportunity
2. Possible competitors
3. Target audience
4. Risks
5. Final quick opinion
"""
    return ask_llm(prompt)

if __name__ == "__main__":
    idea = "AI tutor for Telugu speaking students"
    print("=" * 60)
    print(f"Researching Startup Idea: {idea}")
    print("=" * 60)
    report = startup_research_agent(idea)
    print(report)
