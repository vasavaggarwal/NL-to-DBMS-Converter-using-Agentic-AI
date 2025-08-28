# db/mongo_runner.py
import pandas as pd
from pymongo import MongoClient

# --- 1) Connect to MongoDB (no auth, default localhost:27017) ---
_client = MongoClient("mongodb://127.0.0.1:27017/")
_db = _client["demo_db"]
_collection = _db["customers"]

# --- 2) Run a MongoDB query (find) ---
def run_mongo_query(query: dict, projection: dict = None):
    """
    Run a MongoDB find query and return results as pandas DataFrame.
    :param query: MongoDB filter dict (e.g., {"Age": {"$gt": 30}})
    :param projection: MongoDB projection dict (e.g., {"Age": 1, "Spending_Score": 1})
    """
    cursor = _collection.find(query, projection)
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df

# --- 3) Run a MongoDB aggregation pipeline ---
def run_mongo_aggregate(pipeline: list):
    """
    Run a MongoDB aggregation pipeline and return results as pandas DataFrame.
    :param pipeline: list of MongoDB aggregation stages
    """
    cursor = _collection.aggregate(pipeline)
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df.drop(columns=["_id"], inplace=True)
    return df

# --- 4) Demo (optional manual test) ---
if __name__ == "__main__":
    # Simple filter
    print(run_mongo_query({"Genre": "Male"}).head())

    # Aggregation example
    pipeline = [
        {"$match": {"Genre": "Male"}},
        {"$group": {"_id": None, "avg_income": {"$avg": "$Annual_Income_kUSD"}}}
    ]
    print(run_mongo_aggregate(pipeline))