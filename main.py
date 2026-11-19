"""
Interactive CLI Runner for LangGraph Workflows & AI Agent Concepts
"""

import sys
import os

def display_menu():
    print("\n" + "=" * 65)
    print("   AI AGENT & LANGGRAPH WORKFLOWS MASTER SUITE")
    print("=" * 65)
    print("1. Module 1: Basic Groq LLM Query (01_basic_llm.py)")
    print("2. Module 2: Rule-Based Tool Agent (02_simple_calculator_agent.py)")
    print("3. Module 3: Live Web Research Agent (03_web_search_agent.py)")
    print("4. Module 4: LangGraph Linear Workflow (04_langgraph_linear_workflow.py)")
    print("5. Module 5: LangGraph Conditional Routing (05_langgraph_conditional_workflow.py)")
    print("6. Module 6: LangGraph Human-in-the-Loop (06_langgraph_human_in_the_loop.py)")
    print("7. Custom Run: Validate Your Own Startup Idea with LangGraph")
    print("0. Exit")
    print("=" * 65)

def run_custom():
    idea = input("\nEnter your startup idea: ").strip()
    if not idea:
        print("Startup idea cannot be empty.")
        return
    
    print("\nChoose workflow:")
    print("  [1] Linear Workflow")
    print("  [2] Conditional Routing (by risk level)")
    print("  [3] Human-in-the-Loop Approval")
    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        from importlib import import_module
        mod = import_module("04_langgraph_linear_workflow")
        app = mod.build_linear_graph()
        result = app.invoke({"idea": idea})
        print("\n" + "=" * 60)
        print("REPORT:\n", result["final_report"])
    elif choice == "2":
        from importlib import import_module
        mod = import_module("05_langgraph_conditional_workflow")
        app = mod.build_conditional_graph()
        result = app.invoke({"idea": idea})
        print("\n" + "=" * 60)
        print(f"Risk Level: {result.get('risk_level')}")
        if result.get("improvement"):
            print("\nImprovements:\n", result["improvement"])
        print("\nReport:\n", result["final_report"])
    elif choice == "3":
        from importlib import import_module
        mod = import_module("06_langgraph_human_in_the_loop")
        app = mod.build_hitl_graph()
        result = app.invoke({"idea": idea})
        print("\n" + "=" * 60)
        print("OUTCOME:\n", result["final_report"])
    else:
        print("Invalid workflow selection.")

def main():
    while True:
        display_menu()
        choice = input("Select an option (0-7): ").strip()

        if choice == "0":
            print("\nExiting. Happy building!")
            break
        elif choice == "1":
            import subprocess
            subprocess.run([sys.executable, "01_basic_llm.py"])
        elif choice == "2":
            import subprocess
            subprocess.run([sys.executable, "02_simple_calculator_agent.py"])
        elif choice == "3":
            import subprocess
            subprocess.run([sys.executable, "03_web_search_agent.py"])
        elif choice == "4":
            import subprocess
            subprocess.run([sys.executable, "04_langgraph_linear_workflow.py"])
        elif choice == "5":
            import subprocess
            subprocess.run([sys.executable, "05_langgraph_conditional_workflow.py"])
        elif choice == "6":
            import subprocess
            subprocess.run([sys.executable, "06_langgraph_human_in_the_loop.py"])
        elif choice == "7":
            run_custom()
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
