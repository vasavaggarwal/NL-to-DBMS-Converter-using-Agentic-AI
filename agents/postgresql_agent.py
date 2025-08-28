# agents/postgresql_agent.py
import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from db.query_runner import init_db, run_query

# ---- LLM setup (keep it deterministic) ----
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=api_key,
    temperature=0,
    model_kwargs={"top_p": 0}
)

POSTGRESQL_SCHEMA_AND_RULES = """
DATABASE: demo_db
TABLE: mytable
COLUMNS (EXACT NAMES; case sensitive when quoted):
- CustomerID
- Genre
- Age
- "Annual Income (k$)"
- "Spending Score (1-100)"

CATEGORICAL VALUES:
- Genre must be exactly 'Male' or 'Female' (case-sensitive). Do not change case.

HARD RULES:
- Your output MUST be either:
  1) A single valid PostgreSQL SELECT statement that references ONLY `mytable` and ONLY the columns listed above; or
  2) EXACTLY: INVALID QUERY
- If the user's request mentions, implies, or requires ANY column/attribute not in the list above,
  you MUST output EXACTLY: INVALID QUERY.
- Do NOT guess or map to similar words (e.g., "Gender" is NOT "Genre").
- When referencing columns containing spaces or parentheses, use double quotes,
  e.g., "Annual Income (k$)", "Spending Score (1-100)".
- No comments, no explanations, no JSON—FINAL OUTPUT MUST BE ONLY the SQL or EXACTLY: INVALID QUERY.

EXAMPLE (for casing of Genre only):
-- User: average annual income of female customers
SELECT AVG("Annual Income (k$)") AS avg_income
FROM mytable
WHERE Genre = 'Female';
"""

SYSTEM_PROMPT = f"""You are a PostgreSQL expert agent.
Follow the rules precisely.

{POSTGRESQL_SCHEMA_AND_RULES}
"""

# ---- Single in-memory DuckDB connection on the CSV ----
_CONN = init_db("db/mockdb_1.csv")

def _generate_pg_sql(nl_query: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nUser request: {nl_query}\nFinal output (SQL or EXACTLY 'INVALID QUERY'):"
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()

def run_postgresql_agent(nl_query: str) -> Dict[str, Any]:
    try:
        sql_text = _generate_pg_sql(nl_query)

        if sql_text.upper() == "INVALID QUERY":
            return {"success": False, "error": "error occurred", "data": None}

        df = run_query(_CONN, sql_text, dialect="postgres")
        return {"success": True, "error": None, "data": df}

    except Exception:
        return {"success": False, "error": "error occurred", "data": None}

if __name__ == "__main__":
    print(run_postgresql_agent("Find the number of customers aged below 30"))