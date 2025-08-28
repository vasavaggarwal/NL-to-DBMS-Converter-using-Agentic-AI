# agents/supervisor.py
from __future__ import annotations
from typing import TypedDict, Optional, Literal
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from agents.sql_agent import run_sql_agent
from agents.mysql_agent import run_mysql_agent
from agents.postgresql_agent import run_postgresql_agent
from agents.mongodb_agent import run_mongodb_agent

# ---------------------------
# Model (deterministic)
# ---------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama3-8b-8192", api_key=api_key, temperature=0, model_kwargs={"top_p": 0})

# ---------------------------
# State
# ---------------------------
Route = Literal["SQL", "MySQL", "PostgreSQL", "MongoDB", "REJECT"]

class S(TypedDict, total=False):
    # What the supervisor LLM sees (single combined string):
    user_input: str                 # e.g., "Query: <nl> | Language: <SQL|MySQL|PostgreSQL|MongoDB>"
    # What agents get (no parsing by us; we already have it from UI):
    nl_query: str                   # raw natural-language question only
    # Supervisor routing decision:
    route: Route
    # Agent result:
    result: dict
    language: Optional[str]

# ---------------------------
# Supervisor (LLM decides)
# ---------------------------
SUPERVISOR_PROMPT = """You are a routing supervisor for a multi-DBMS demo.

You receive ONE input string that contains:
- a natural-language question ("Query: ...")
- and a human-selected DBMS label ("Language: SQL|MySQL|PostgreSQL|MongoDB").

Your job:
- Decide which SINGLE agent to call based on the language portion.
- Output EXACTLY one token on a single line:
  SQL
  MySQL
  PostgreSQL
  MongoDB
If the input is unusable, output EXACTLY: REJECT.

IMPORTANT:
- Output ONLY the token. No punctuation, commentary, code blocks, or extra text.
"""

def supervisor_decider(state: S) -> S:
    text = state.get("user_input", "")
    prompt = f"{SUPERVISOR_PROMPT}\n\nINPUT:\n{text}\n\nROUTE:"
    out = (llm.invoke(prompt).content or "").strip()
    state["route"] = out  # trust the LLM
    return state

# ---------------------------
# Agent nodes (use nl_query only)
# ---------------------------
def node_sql(state: S) -> S:
    return {"result": run_sql_agent(state["nl_query"]), "language": "SQL"}

def node_mysql(state: S) -> S:
    return {"result": run_mysql_agent(state["nl_query"]), "language": "MySQL"}

def node_postgres(state: S) -> S:
    return {"result": run_postgresql_agent(state["nl_query"]), "language": "PostgreSQL"}

def node_mongo(state: S) -> S:
    return {"result": run_mongodb_agent(state["nl_query"]), "language": "MongoDB"}

# ---------------------------
# Router (conditional edges only)
# ---------------------------
def route_from_super(state: S) -> str:
    token = (state.get("route") or "").strip()
    return {
        "SQL": "sql",
        "MySQL": "mysql",
        "PostgreSQL": "postgres",
        "MongoDB": "mongo",
        "REJECT": END,
    }.get(token, END)

# ---------------------------
# Build graph
# ---------------------------
def build_graph():
    g = StateGraph(S)
    g.add_node("supervisor", supervisor_decider)
    g.add_node("sql", node_sql)
    g.add_node("mysql", node_mysql)
    g.add_node("postgres", node_postgres)
    g.add_node("mongo", node_mongo)

    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor", route_from_super, {
        "sql": "sql",
        "mysql": "mysql",
        "postgres": "postgres",
        "mongo": "mongo",
        END: END
    })

    # IMPORTANT: end after agent runs (no loop-back to supervisor)
    for n in ("sql", "mysql", "postgres", "mongo"):
        g.add_edge(n, END)

    return g.compile()  # stateless; no checkpointer requirement

app = build_graph()

# ---------------------------
# Public helper
# ---------------------------
def run_supervisor(query: str, language: str) -> dict:
    """
    UI should call this with the raw NL query and the dropdown language.
    We hand the combined string ONLY to the supervisor model.
    Agents receive the raw NL query via state (no parsing).
    """
    combined = f"Query: {query} | Language: {language}"
    final = app.invoke({"user_input": combined, "nl_query": query})

    # If routing failed:
    if final.get("route") in (None, "REJECT"):
        return {
            "success": False,
            "error": "Error, please enter one of the following languages: SQL, MySQL, PostgreSQL, MongoDB",
            "data": None,
            "language": None,
        }

    out = final.get("result", {}) or {}
    out["language"] = final.get("language", final.get("route"))
    return out

# ---------------------------
# Quick CLI test
# ---------------------------
if __name__ == "__main__":
    print(run_supervisor("List all female customers.", "SQL"))
    print(run_supervisor("Show customers older than 40.", "MySQL"))
    print(run_supervisor("Find the customers from Country = India", "PostgreSQL"))
    print(run_supervisor("Show customers with a spending score greater than 80.", "MongoDB"))
