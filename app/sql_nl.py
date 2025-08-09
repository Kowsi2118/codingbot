from __future__ import annotations

from typing import Dict, List, Optional

import duckdb
import pandas as pd

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


SYSTEM_PROMPT = (
    "You are an expert data analyst. Generate a single, correct SQL query for DuckDB. "
    "Only output the SQL, no explanation. Use the table named data."
)


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def get_schema_ddl(df: pd.DataFrame) -> str:
    cols = []
    for name, dtype in df.dtypes.items():
        if str(dtype).startswith("int"):
            t = "INTEGER"
        elif str(dtype).startswith("float"):
            t = "DOUBLE"
        elif "datetime" in str(dtype):
            t = "TIMESTAMP"
        else:
            t = "TEXT"
        cols.append(f"{name} {t}")
    return "CREATE TABLE data (" + ", ".join(cols) + ");"


def generate_sql_from_question(
    question: str,
    df: pd.DataFrame,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> str:
    schema = get_schema_ddl(df)

    if openai_api_key and OpenAI is not None:
        client = OpenAI(api_key=openai_api_key)
        prompt = (
            f"Schema:\n{schema}\n\nQuestion: {question}\n\n"
            "Return only the SQL. Use functions supported by DuckDB."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            sql = resp.choices[0].message.content.strip()  # type: ignore
            # Remove code fences if any
            if sql.startswith("```) ") or sql.startswith("```"):
                sql = sql.strip("`\n ")
            return sql
        except Exception:
            pass

    # Fallback simple heuristic SQL
    # Provide a simple default such as descriptive stats
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        col = num_cols[0]
        return f"SELECT COUNT(*) AS n, AVG({col}) AS avg_{col}, STDDEV_SAMP({col}) AS sd_{col} FROM data;"
    else:
        return "SELECT COUNT(*) AS n FROM data;"


def run_sql(sql: str, df: pd.DataFrame) -> pd.DataFrame:
    con = get_duckdb_connection()
    con.register("data", df)
    return con.execute(sql).df()