# agents/mongodb_agent.py
import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Runner funcs (PyMongo-backed; no init needed)
from db.mongo_runner import run_mongo_query, run_mongo_aggregate

# LLM setup
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama3-8b-8192",
    api_key=api_key,
    temperature=0,
    model_kwargs={"top_p": 0},
)

MONGO_SCHEMA_AND_RULES = """
DATABASE: demo_db
COLLECTION: customers
DOCUMENT FIELDS (EXACT NAMES; exact casing):
- CustomerID: string (e.g., "0001")
- Genre: string ("Male" or "Female")
- Age: integer
- Annual_Income_kUSD: integer
- Spending_Score: integer

OUTPUT FORMAT (choose exactly ONE):
1) EXACTLY: INVALID QUERY
2) Simple find:
   {"filter": <MongoDB filter dict>, "projection": <optional projection dict>}
3) Aggregation pipeline:
   {"aggregate": [ <MongoDB pipeline array> ]}

DECISION POLICY (intent → output type):
- If the user asks for a single number or grouped numbers (average/mean, count/how many, sum/total, min/youngest/lowest, max/oldest/highest, median, std dev) or says “by <field>”, return an aggregation pipeline with $match (optional) then $group (and optionally $project/$sort/$limit).
- If the user asks for “top/most/bottom/least N”, return an aggregation pipeline with $sort then $limit.
- If the user asks to list/show/find records that meet conditions, return a simple find.
- Use ONLY the five allowed fields. Otherwise, EXACTLY: INVALID QUERY.
- Valid ops only: $gt, $gte, $lt, $lte, $in, $and, $or, $regex, $match, $group, $project, $sort, $limit, $count, $avg, $min, $max, $sum.
- Categorical values are case-sensitive ('Male', 'Female').
- Final output MUST be ONLY the JSON object or EXACTLY: INVALID QUERY.
"""

SYSTEM_PROMPT = f"""You are a MongoDB expert agent.
Follow the rules precisely.

{MONGO_SCHEMA_AND_RULES}
"""

def _generate_mongo_json(nl_query: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\nUser request: {nl_query}\nFinal output (JSON or EXACTLY 'INVALID QUERY'):"
    resp = llm.invoke(prompt)
    return (resp.content or "").strip()

def run_mongodb_agent(nl_query: str) -> Dict[str, Any]:
    try:
        out = _generate_mongo_json(nl_query)

        if out.upper() == "INVALID QUERY":
            return {"success": False, "error": "error occurred", "data": None}

        obj = json.loads(out)  # let it raise on malformed; caught below

        # If aggregation provided, run it
        if isinstance(obj, dict) and "aggregate" in obj:
            pipeline = obj["aggregate"]
            df = run_mongo_aggregate(pipeline)
            return {"success": True, "error": None, "data": df}

        # Otherwise assume simple find
        if isinstance(obj, dict) and "filter" in obj:
            flt = obj.get("filter", {})
            proj: Optional[dict] = obj.get("projection")
            df = run_mongo_query(flt, projection=proj)
            return {"success": True, "error": None, "data": df}

        return {"success": False, "error": "error occurred", "data": None}

    except Exception:
        return {"success": False, "error": "error occurred", "data": None}

if __name__ == "__main__":
    # Your test case:
    print(run_mongodb_agent("Find customers with country = India"))