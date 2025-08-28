# agents/sql_agent.py
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from db.query_runner import init_db, run_query

# ---- LLM setup (deterministic; no warnings about top_p) ----
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=api_key,
    temperature=0,
    model_kwargs={"top_p": 0},
)

# ---- Strict schema + refusal rules (no examples) ----
SQL_SCHEMA_AND_RULES = """
DATABASE: demo_db
TABLE: mytable
COLUMNS (EXACT NAMES; case sensitive when quoted):
- CustomerID
- Genre
- Age
- "Annual Income (k$)"
- "Spending Score (1-100)"

HARD RULES:
- Your output MUST be either:
  1) A single valid standard SQL SELECT statement that references ONLY `mytable` and ONLY the columns listed above; or
  2) EXACTLY: INVALID QUERY
- If the user's request mentions, implies, or requires ANY column/attribute not in the list above,
  you MUST output EXACTLY: INVALID QUERY.
- Do NOT guess or map to similar words (e.g., "Gender" is NOT "Genre").
- When referencing columns containing spaces or parentheses, use double quotes,
  e.g., "Annual Income (k$)", "Spending Score (1-100)".
- No comments, no explanations, no JSON—FINAL OUTPUT MUST BE ONLY the SQL or EXACTLY: INVALID QUERY.
"""

SYSTEM_PROMPT = f"""You are an SQL expert agent.
Follow the rules precisely.

{SQL_SCHEMA_AND_RULES}
"""

# ---- Single in-memory DuckDB connection on the CSV ----
_CONN = init_db("db/mockdb_1.csv")

def _generate_sql(nl_query: str) -> str:
    # Minimal, strict prompting. Assistant must return ONLY SQL or INVALID QUERY.
    prompt = f"{SYSTEM_PROMPT}\nUser request: {nl_query}\nFinal output (SQL or EXACTLY 'INVALID QUERY'):"
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()

def run_sql_agent(nl_query: str) -> Dict[str, Any]:
    """
    Returns exactly:
      - success: bool
      - error: str | None         (generic 'error occurred' only)
      - data: pandas.DataFrame | None
    """
    try:
        sql_text = _generate_sql(nl_query)

        # Model-driven refusal—no extra validation code
        if sql_text.upper() == "INVALID QUERY":
            return {"success": False, "error": "error occurred", "data": None}

        # Execute on DuckDB (standard SQL runs fine; we treat it as duckdb dialect)
        df = run_query(_CONN, sql_text, dialect="duckdb")
        return {"success": True, "error": None, "data": df}

    except Exception:
        # Keep it generic; no white-boxing
        return {"success": False, "error": "error occurred", "data": None}

# Quick manual test (optional)
if __name__ == "__main__":
    print(run_sql_agent("Find the average annual income of male customers."))