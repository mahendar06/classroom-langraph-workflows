"""
Module 5: LangGraph Conditional Routing Workflow
Demonstrates dynamic branch routing in LangGraph based on runtime evaluation:
1. Research Node: Researches market trends.
2. Risk Analysis Node: Pinpoints key risks.
3. Risk Level Classifier Node: Classifies overall risk as LOW, MEDIUM, or HIGH.
4. Conditional Router:
   - If risk is HIGH -> routes to Improvement Node before Final Report.
   - Otherwise -> routes directly to Final Report Node.
"""

import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, END

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
            {"role": "system", "content": "You are a professional venture capital advisor and risk assessment expert."},
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
            results.append({"title": "Offline fallback", "href": "", "body": f"Live search unavailable ({e})."})
    return results

def startup_research_agent(idea: str) -> str:
    search_data = web_search(f"{idea} market trends competitors", max_results=5)
    context = "\n\n".join([
        f"Title: {item['title']}\nSummary: {item['body']}\nURL: {item['href']}"
        for item in search_data
    ])
    prompt = f"""
You are a startup research analyst.
Startup idea: {idea}
Web research data:
{context}

Provide market opportunity, competitors, target audience, risks, and quick opinion.
"""
    return ask_llm(prompt)

# State definition with dynamic branching fields
class StartupState2(TypedDict):
    idea: str
    research: str
    risks: str
    risk_level: str
    improvement: str
    final_report: str

# Node Functions
def research_node(state: StartupState2):
    print("-> [Node 1] Performing market research...")
    research = startup_research_agent(state["idea"])
    return {"research": research}

def risk_node(state: StartupState2):
    print("-> [Node 2] Analyzing startup risks...")
    prompt = f"""
Analyze risks for this startup idea.
Idea: {state['idea']}
Research: {state['research']}

Give top 5 risks with clear explanation.
"""
    risks = ask_llm(prompt)
    return {"risks": risks}

def risk_level_node(state: StartupState2):
    print("-> [Node 3] Evaluating risk level classification...")
    prompt = f"""
Based on the risks below, classify risk level as only one word: LOW, MEDIUM, or HIGH.

Risks:
{state['risks']}
"""
    risk_level = ask_llm(prompt).strip().upper()
    return {"risk_level": risk_level}

def improvement_node(state: StartupState2):
    print("-> [Node 4 - Conditional Branch] High Risk Detected! Generating mitigation strategies...")
    prompt = f"""
The startup idea has been classified as high risk.

Idea:
{state['idea']}

Risks:
{state['risks']}

Suggest concrete strategic improvements and pivot opportunities to mitigate these risks.
"""
    improvement = ask_llm(prompt)
    return {"improvement": improvement}

def report_node(state: StartupState2):
    print("-> [Node 5] Generating final validation report...")
    improvement_context = state.get("improvement", "None required (Low/Medium Risk).")
    prompt = f"""
Create final startup validation report.

Idea: {state['idea']}
Research: {state['research']}
Risks: {state['risks']}
Assessed Risk Level: {state.get('risk_level', 'N/A')}
Suggested Improvements (if any): {improvement_context}

Format:
- Idea Summary
- Market Opportunity
- Risk Assessment (Level: {state.get('risk_level', 'N/A')})
- Mitigation & Improvement Strategy
- Recommendation & Final Score out of 10
"""
    final_report = ask_llm(prompt)
    return {"final_report": final_report}

# Conditional routing decision function
def route_by_risk(state: StartupState2) -> str:
    risk = state.get("risk_level", "")
    print(f"--> Router evaluating: Detected Risk Level = '{risk}'")
    if "HIGH" in risk:
        return "improvement"
    return "final_report"

def build_conditional_graph():
    graph = StateGraph(StartupState2)

    # Register Nodes
    graph.add_node("research", research_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("risk_level", risk_level_node)
    graph.add_node("improvement", improvement_node)
    graph.add_node("final_report", report_node)

    # Edges
    graph.set_entry_point("research")
    graph.add_edge("research", "risk_analysis")
    graph.add_edge("risk_analysis", "risk_level")

    # Add Conditional Edges
    graph.add_conditional_edges(
        "risk_level",
        route_by_risk,
        {
            "improvement": "improvement",
            "final_report": "final_report"
        }
    )

    graph.add_edge("improvement", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_conditional_graph()
    sample_idea = "AI tutor for Telugu speaking students"
    print("=" * 60)
    print(f"Starting Conditional LangGraph Workflow for: {sample_idea}")
    print("=" * 60)
    result = app.invoke({"idea": sample_idea})
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY:")
    print("=" * 60)
    print(f"Classified Risk Level: {result.get('risk_level')}")
    if result.get("improvement"):
        print("\n[Mitigation Strategies Generated]:")
        print(result["improvement"])
    print("\n[Final Report]:")
    print(result["final_report"])
