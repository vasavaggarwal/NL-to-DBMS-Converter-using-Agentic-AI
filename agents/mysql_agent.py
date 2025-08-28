# agents/mysql_agent.py
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# DuckDB + sqlglot runner on your CSV
from db.query_runner import init_db, run_query

# --- 1) LLM setup ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=api_key,
    temperature=0
)

# --- 2) Prompt exactly matches your CSV-backed table ---
# NOTE: table name is `mytable` (as registered in query_runner.init_db),
# and column names are exactly as in mockdb_1.csv.
MYSQL_SCHEMA_AND_RULES = """
DATABASE: demo_db
TABLE: mytable
COLUMNS (EXACT NAMES; use backticks when they contain spaces/parentheses):
- CustomerID
- Genre
- Age
- `Annual Income (k$)`
- `Spending Score (1-100)`

HARD RULES:
- Your output MUST be either:
  1) A single valid MySQL SELECT statement that references ONLY `mytable` and ONLY the columns listed above; or
  2) EXACTLY: INVALID QUERY
- If the user's request mentions, implies, or requires ANY column/attribute not in the list above,
  you MUST output EXACTLY: INVALID QUERY.
- Do NOT guess or map to similar words (e.g., "Gender" is NOT "Genre").
- When referencing columns containing spaces or parentheses, use backticks,
  e.g., `Annual Income (k$)`, `Spending Score (1-100)`.
- No comments, no explanations, no JSON — FINAL OUTPUT MUST BE ONLY the SQL or EXACTLY: INVALID QUERY.
"""

SYSTEM_PROMPT = f"""You are a MySQL expert agent.
Follow the rules precisely.

{MYSQL_SCHEMA_AND_RULES}
"""
# --- 3) Keep a single in-memory connection for speed ---
_CONN = init_db("db/mockdb_1.csv")

def _generate_mysql_sql(nl_query: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nUser query: {nl_query}\nSQL:"
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()

def run_mysql_agent(nl_query: str) -> Dict[str, Any]:
    """
    Returns exactly:
      - success: bool
      - error: str | None   (generic 'error occurred' when anything goes wrong)
      - data: pandas.DataFrame | None
    """
    try:
        sql_text = _generate_mysql_sql(nl_query)

        # If agent flags invalid, do NOT execute
        if sql_text.upper() == "INVALID QUERY":
            return {"success": False, "error": "Error Occurred", "data": None}

        # Execute MySQL-dialect SQL on the CSV via DuckDB (sqlglot handles translation)
        df = run_query(_CONN, sql_text, dialect="mysql")
        return {"success": True, "error": None, "data": df}

    except Exception:
        # Hide internals/translation details as requested
        return {"success": False, "error": "error occurred", "data": None}

# Quick manual test (optional)
if __name__ == "__main__":
    print(run_mysql_agent("Find the average annual income of male customers."))
