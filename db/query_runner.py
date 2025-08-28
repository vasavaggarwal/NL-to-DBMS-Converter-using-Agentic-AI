# db/query_runner.py
import duckdb
import pandas as pd
import sqlglot

# Load CSV into a DuckDB in-memory table
def init_db(csv_path="db/mockdb_1.csv"):
    # Read CSV into pandas
    df = pd.read_csv(csv_path)

    # Create in-memory duckdb connection
    conn = duckdb.connect(database=":memory:")
    conn.register("mytable", df)  # register pandas df as table "mytable"
    return conn

# Translate query into DuckDB-compatible SQL
def translate_query(query: str, dialect: str) -> str:
    """
    dialect can be: 'mysql', 'postgres', 'sqlite', 'duckdb'
    """
    try:
        translated = sqlglot.transpile(query, read=dialect, write="duckdb")[0]

        return translated
    except Exception as e:
        raise ValueError(f"Could not translate query: {e}")

# Execute query on DuckDB
def run_query(conn, query: str, dialect: str = "duckdb"):
    # Convert MySQL/Postgres -> DuckDB SQL
    if dialect != "duckdb":
        query = translate_query(query, dialect)

    # Execute query
    result_df = conn.execute(query).fetchdf()
    return result_df

