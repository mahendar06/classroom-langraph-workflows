"""
Module 6: LangGraph Human-in-the-Loop Workflow
Demonstrates integrating human governance and review into an AI workflow:
1. Research Node: Scrapes market trends.
2. Risk Analysis Node: Highlights risks.
3. Human Approval Node: Halts autonomous execution to prompt human operator.
4. Conditional Router:
   - If human approves ("yes") -> Proceeds to generate final validation report.
   - If human rejects ("no") -> Diverts to rejection node and prompts idea refinement.
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
            {"role": "system", "content": "You are a startup evaluation specialist."},
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
Idea: {idea}
Web research data:
{context}

Provide market opportunity, competitors, target audience, risks, and quick opinion.
"""
    return ask_llm(prompt)

# State Definition
class StartupState3(TypedDict):
    idea: str
    research: str
    risks: str
    approval: str
    final_report: str

def research_node(state: StartupState3):
    print("-> [Node 1] Gathering market insights...")
    research = startup_research_agent(state["idea"])
    return {"research": research}

def risk_node(state: StartupState3):
    print("-> [Node 2] Evaluating venture risks...")
    prompt = f"""
Analyze risks for this startup idea.
Idea: {state['idea']}
Research: {state['research']}

Give top 5 risks with explanation.
"""
    risks = ask_llm(prompt)
    return {"risks": risks}

def human_approval_node(state: StartupState3):
    print("\n" + "=" * 25 + " HUMAN REVIEW GATE " + "=" * 25)
    print(f"Startup Idea: {state['idea']}")
    print("-" * 70)
    print("Identified Risks:")
    print(state["risks"])
    print("=" * 70)

    # In non-interactive environments, allow fallback or prompt
    auto_choice = os.getenv("HITL_AUTO_APPROVE")
    if auto_choice:
        approval = auto_choice
        print(f"[Auto-Response]: {approval}")
    else:
        try:
            approval = input("\nDo you approve generating the final report? (yes/no): ").strip().lower()
        except EOFError:
            approval = "yes"

    if approval in ("yes", "y"):
        return {"approval": "approved"}
    else:
        return {"approval": "rejected"}

def report_node(state: StartupState3):
    print("-> [Node 4A] Human approved! Generating final validation report...")
    prompt = f"""
Create final startup validation report.

Idea: {state['idea']}
Research: {state['research']}
Risks: {state['risks']}

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

def rejected_node(state: StartupState3):
    print("-> [Node 4B] Report generation rejected by human operator.")
    return {
        "final_report": (
            "Human operator rejected final report generation. "
            "Please review and mitigate identified risks before requesting re-evaluation."
        )
    }

def approval_router(state: StartupState3) -> str:
    decision = state.get("approval")
    print(f"--> Approval Router: Decision is '{decision}'")
    if decision == "approved":
        return "final_report"
    return "rejected"

def build_hitl_graph():
    graph = StateGraph(StartupState3)

    # Add Nodes
    graph.add_node("research", research_node)
    graph.add_node("risk_analysis", risk_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("final_report", report_node)
    graph.add_node("rejected", rejected_node)

    # Entry point & Linear edges
    graph.set_entry_point("research")
    graph.add_edge("research", "risk_analysis")
    graph.add_edge("risk_analysis", "human_approval")

    # Conditional human approval branch
    graph.add_conditional_edges(
        "human_approval",
        approval_router,
        {
            "final_report": "final_report",
            "rejected": "rejected"
        }
    )

    graph.add_edge("final_report", END)
    graph.add_edge("rejected", END)

    return graph.compile()

if __name__ == "__main__":
    app = build_hitl_graph()
    sample_idea = "AI tutor for Telugu speaking students"
    print("=" * 60)
    print(f"Starting Human-In-The-Loop Workflow for: {sample_idea}")
    print("=" * 60)
    result = app.invoke({"idea": sample_idea})
    print("\n" + "=" * 60)
    print("FINAL OUTCOME:")
    print("=" * 60)
    print(result["final_report"])
