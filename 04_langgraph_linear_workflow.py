"""
Module 4: LangGraph Linear Workflow
Demonstrates a multi-step sequential state pipeline using LangGraph:
1. Research Node: Gathers web data & drafts market analysis.
2. Risk Analysis Node: Assesses top 5 critical risks from the research.
3. Final Report Node: Synthesizes validation report with score out of 10.
"""

import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

# Optional fallback search
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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert startup advisor and market research analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content

def web_search(query: str, max_results: int = 5):
    results = []
    if DDGS:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    })
        except Exception as e:
            results.append({"title": "Offline fallback", "href": "", "body": f"Note: Live search unavailable ({e})."})
    return results

def startup_research_agent(idea: str) -> str:
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

Give:
1. Market opportunity
2. Possible competitors
3. Target audience
4. Risks
5. Final quick opinion
"""
    return ask_llm(prompt)

# State Definition
class StartupState(TypedDict):
    idea: str
    research: str
    risks: str
    final_report: str

# Graph Node Functions
def research_node(state: StartupState):
    print("-> Running Node: research_node")
    research = startup_research_agent(state["idea"])
    return {"research": research}

def risk_node(state: StartupState):
    print("-> Running Node: risk_node")
    prompt = f"""
Analyze risks for this startup idea.

Idea:
{state['idea']}

Research:
{state['research']}

Give top 5 risks with clear explanation.
"""
    risks = ask_llm(prompt)
    return {"risks": risks}

def report_node(state: StartupState):
    print("-> Running Node: report_node")
    prompt = f"""
Create final startup validation report.

Idea:
{state['idea']}

Research:
{state['research']}

Risks:
{state['risks']}

Format:
- Idea Summary
- Market Opportunity
- Risks
- Recommendation  
- Score out of 10
- Action Plan
"""
    final_report = ask_llm(prompt)
    return {"final_report": final_report}

def build_linear_graph():
    graph = StateGraph(StartupState)

    # Register Nodes
    graph.add_node("research", research_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("final_report", report_node)

    # Define Linear Edges
    graph.set_entry_point("research")
    graph.add_edge("research", "risk_analysis")
    graph.add_edge("risk_analysis", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_linear_graph()
    sample_idea = "AI tutor for Telugu speaking students"
    print("=" * 60)
    print(f"Starting Linear LangGraph Execution for: {sample_idea}")
    print("=" * 60)
    result = app.invoke({"idea": sample_idea})
    print("\n" + "=" * 60)
    print("FINAL VALIDATION REPORT:")
    print("=" * 60)
    print(result["final_report"])
