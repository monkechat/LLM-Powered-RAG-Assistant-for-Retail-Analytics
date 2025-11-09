import os, re, textwrap
import pandas as pd
from databricks import sql
from openai import OpenAI
from pyspark.sql.utils import AnalysisException, ParseException
from typing import List, Dict, Set
import vector_db

# Initialize the variables by calling the function from vector_db module
schema_retriever, schema_child_vs, joins_vs = vector_db.initialize_variables()

def get_data_from_query(query: str,
                        sql_dialect: str = "Spark SQL (Databricks SQL)",
                        first_model: str = "gpt-4o-mini"):

    # 1) First attempt (gpt-4o-mini)
    sql = generate_sql_from_query(query, sql_dialect = sql_dialect, model = first_model)
    print(f"Generated SQL. Running first attempt...\n{sql}\n")
    try:
        return fetch_data(sql) # success
    except (AnalysisException, ParseException, Exception) as e1:
        err1 = concise_spark_error(e1)

        # print or log if you want visibility
        print(f"[Attempt 1 FAILED] {err1}\n")

    # 2) First repair with GPT-4o
    repaired_sql_1 = repair_sql_with_error(query, bad_sql=sql, error_text=err1,
                                           sql_dialect=sql_dialect, model="gpt-4o")
    print(f"Running second attempt with repaired SQL...\n{repair_sql_with_error}\n")
    try:
        return fetch_data(repaired_sql_1) # success
    except (AnalysisException, ParseException, Exception) as e2:
        err2 = concise_spark_error(e2)
        print(f"[Attempt 2 FAILED] {err2}\n")

    # 3) Second repair with GPT-4o (last try)
    repaired_sql_2 = repair_sql_with_error(query, bad_sql=repaired_sql_1, error_text=err2,
                                           sql_dialect=sql_dialect, model="gpt-4o")
    print(f"Running third attempt with repaired SQL...\n{repaired_sql_2}\n")
    try:
        return fetch_data(repaired_sql_2) # success
    except (AnalysisException, ParseException, Exception) as e3:
        err3 = concise_spark_error(e3)

        print(f"[Attempt 3 FAILED] {err3}\n")

        return 'Error'

def generate_sql_from_query(user_query: str,
                            sql_dialect: str = "PostgreSQL",
                            k_schema: int = 6,
                            k_joins_per_side: int = 6,
                            model: str = "gpt-4o-mini",
                            temperature: float = 0.0,
                            max_tokens: int = 800) -> str:
    """
    Full pipeline:
      1) retrieval (parents + joins)
      2) prompt assembly
      3) LLM call (GPT-4o-mini)
      4) post-process to ensure SQL-only (strip code fences)
    """
    # --- 1) Retrieval ---
    ctx = retrieve_context(
        user_query,
        k_schema=k_schema,
        k_joins_per_side=k_joins_per_side
    )

    # --- 2) Build messages ---
    messages = _make_messages(user_query, ctx, sql_dialect=sql_dialect)

    # --- 3) Call OpenAI ---
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    sql = resp.choices[0].message.content.strip()

    # --- 4) Sanitize: strip any accidental code fences/backticks ---
    sql = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", sql, flags=re.IGNORECASE | re.DOTALL).strip()
    return sql

def retrieve_context(query: str,
                     k_schema: int = 6,
                     k_joins_per_side: int = 6) -> Dict[str, List]:
    """
    High-level retrieval:
      1) schema parents,
      2) candidate tables,
      3) join edges filtered to those tables.
    """
    parents = retrieve_schema_parents(query, k=k_schema)
    tables  = collect_candidate_tables(parents)
    joins   = retrieve_joins_for_tables(query, tables, k_per_side=k_joins_per_side)
    return {"parents": parents, "tables": tables, "joins": joins}

def retrieve_schema_parents(query: str, k: int = 6) -> List:
    """
    Retrieve relevant schema PARENTS via ParentDocumentRetriever.
    (k is controlled by schema_retriever.search_kwargs but we keep it as a semantic arg here.)
    """
    
    # ParentDocumentRetriever returns parents directly
    return schema_retriever.get_relevant_documents(query)

def collect_candidate_tables(_parent_docs: List) -> List[str]:
    """Unique, order-preserving list of table names from retrieved parents."""
    seen: Set[str] = set()
    ordered: List[str] = []
    for d in _parent_docs:
        tbl = _extract_table_name(d)
        if tbl and tbl not in seen:
            seen.add(tbl)
            ordered.append(tbl)
    return ordered

def _extract_table_name(_doc) -> str:
    """Prefer metadata.table; fallback to parsing 'Table: ...' from content."""
    t = _doc.metadata.get("table")
    if t:
        return t.strip()
    
    # fallback parse
    m = re.search(r"^Table:\s*([^\n\r]+)", _doc.page_content, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""

def retrieve_joins_for_tables(query: str,
                              candidate_tables: List[str],
                              k_per_side: int = 6) -> List:
    """
    Retrieve join edges (atomic docs) filtered to the candidate tables.
    We run two filtered searches (from_table and to_table) and dedupe.
    """
    if not candidate_tables:
        return []

    # Search joins relevant to the query but restricted by metadata filters
    hits_from = joins_vs.similarity_search(
        query,
        k = k_per_side,
        filter={"from_table": {"$in": candidate_tables}}
    )
    hits_to = joins_vs.similarity_search(
        query,
        k=k_per_side,
        filter={"to_table": {"$in": candidate_tables}}
    )

    # Deduplicate by (from_table, from_col, to_table, to_col)
    def _key(d):
        m = d.metadata
        return (m.get("from_table",""), m.get("from_col",""),
                m.get("to_table",""),   m.get("to_col",""))

    dedup = {}
    for d in hits_from + hits_to:
        dedup[_key(d)] = d
    return list(dedup.values())

def _make_messages(user_query: str, _ctx: Dict[str, List], sql_dialect: str = "Spark SQL (Databricks SQL)") -> list:
    """
    Build system + user messages. System forces SQL-only output.
    User message contains task + allowed tables + join rules + schema snippets.
    """
    tables = _ctx["tables"]
    joins  = _ctx["joins"]
    parents = _ctx["parents"]

    joins_txt   = _format_joins(joins)
    parents_txt = _format_parents(parents)

    system_msg = textwrap.dedent(f"""
    You are an expert {sql_dialect} SQL generator for a retail analytics dataset (Olist).
    Your job: produce EXACTLY ONE SQL query that answers the user's request.
    HARD RULES:
    - OUTPUT: Return ONLY the SQL. No prose. No backticks. No comments except inline SQL comments if absolutely necessary.
    - SCOPE: Use ONLY the tables, columns, and joins provided in the context below. Do not invent tables/columns.
    - CUSTOMER: always use customer_unique_id (NOT customer_id) for analysis, use customer_id ONLY for JOIN
    - DERIVATIONS:
        * Item-level sales amount must be derived from available columns. For order items, use (price + freight_value).
        * Order-level totals are derived from item rows via SUM(price + freight_value) grouped by order_id (per notes).
    - STRUCTURE: You MAY use CTEs if needed, but the query must culminate in ONE final SELECT statement.
    - DIALECT: Use {sql_dialect} syntax. Do not use DDL or DML. SELECT queries only.
    - JOINS: Use only the allowed join keys shown. Prefer INNER JOIN unless the note implies optional rows (then LEFT JOIN).
    - QUALIFY: Always qualify columns with table aliases if multiple tables are used.
    - AGGREGATION: If you use aggregates, include appropriate GROUP BY for all non-aggregated selected columns.
    - TIME & NULLS: Be explicit about filters. Handle NULLs safely (e.g., COALESCE where needed).
    - LIMITS: Do not add LIMIT unless explicitly asked.
    - If the request is ambiguous, choose the most standard interpretation consistent with the context.
    """)

    user_msg = textwrap.dedent(f"""
    TASK:
    {user_query}

    ALLOWED TABLES (from retrieved parents):
    {', '.join(tables) if tables else '(none)'}

    ALLOWED JOINS (use exactly these keys):
    {joins_txt}

    RELEVANT SCHEMAS (snippets):
    {parents_txt}
    """)

    return [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()}
    ]

def _format_joins(_joins_docs: List) -> str:
    lines = []
    for d in _joins_docs[:12]:  # cap for brevity
        m = d.metadata
        ft, fc = m.get("from_table",""), m.get("from_col","")
        tt, tc = m.get("to_table",""),   m.get("to_col","")
        note   = m.get("note","")
        if ft and fc and tt and tc:
            line = f"- {ft}.{fc} = {tt}.{tc}" + (f"   -- {note}" if note else "")
            lines.append(line)
        else:
            # fallback to raw line if metadata missing
            lines.append("- " + d.page_content.strip())
    return "\n".join(lines) if lines else "(none)"

def _format_parents(_parents: List, max_chars_per_parent: int = 1200, max_total_chars: int = 7000) -> str:
    """Compact, readable context. We cap sizes to keep the prompt lean."""
    chunks = []
    total = 0
    for p in _parents:
        table_name = p.metadata.get("table", "UNKNOWN_TABLE")
        content = p.page_content.strip()
        content = content[:max_chars_per_parent]
        block = f"### TABLE: {table_name}\n{content}\n"
        if total + len(block) > max_total_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks)

def fetch_data(query):
    # Initialize the databricks PAT
    databricks_pat = "YOUR_DATABRICKS_PAT"
    
    # Databricks connection parameters
    server_hostname = "YOUR_SERVER_HOSTNAME"
    http_path = "YOUR_HTTP_PATH"
    
    # Connect to Databricks using the token
    connection = sql.connect(
        server_hostname = server_hostname,
        http_path = http_path,
        access_token = databricks_pat,
    )
    
    # Execute query and fetch data
    cursor = connection.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    # Get column names
    column_names = [desc[0] for desc in cursor.description]
    
    # Convert to DataFrame
    func_df = pd.DataFrame(results, columns = column_names)
    
    # Return dataframe
    return func_df

# ---------- concise Spark error (trim the wall of text) ----------
def concise_spark_error(e: Exception) -> str:
    msg = getattr(e, "desc", None)
    msg = (msg if isinstance(msg, str) and msg.strip() else str(e)).split("JVM stacktrace:")[0].strip()
    first = msg.splitlines()[0].strip() if msg else "Spark error"
    suggestion = next((ln.strip() for ln in msg.splitlines() if "Did you mean" in ln), None) # keep "Did you mean..." if present
    m = re.search(r"line\s+(\d+)\s+pos\s+(\d+)", msg, flags=re.IGNORECASE) # try to capture "line X, pos Y"
    loc = f"line {m.group(1)}, pos {m.group(2)}" if m else None
    parts = [first]
    if suggestion: parts.append(suggestion)
    if loc: parts.append(loc)
    return " | ".join(parts)

# ---------- single-shot repair with GPT-4o given the error + previous SQL ----------
def repair_sql_with_error(user_query: str,
                          bad_sql: str,
                          error_text: str,
                          sql_dialect: str = "Spark SQL (Databricks SQL)",
                          model: str = "gpt-4o",
                          temperature: float = 0.0,
                          max_tokens: int = 900) -> str:
    """
    Rebuild the same retrieval context, show the previous SQL + Spark error,
    and ask GPT-4o to output a corrected SQL (SQL only).
    """
    # Recreate context for transparency & guardrails
    ctx = retrieve_context(user_query)

    # Base messages (same context you used originally)
    messages = _make_messages(user_query, ctx, sql_dialect = sql_dialect)
    
    # Add targeted correction instruction
    correction = f"""
    VALIDATION / RUNTIME ERROR:
    {error_text}

    PREVIOUS_SQL (to fix, do not repeat mistakes):
    {bad_sql}

    REQUIREMENTS:
    - Output ONLY the corrected {sql_dialect} query (no prose, no backticks).
    - Use ONLY tables/columns/joins from the provided context.
    - If a sales metric is needed, derive it from available columns (e.g., SUM(price + freight_value) from order_items).
    - For monthly buckets in Spark, prefer date_trunc('month', <timestamp>).
    - For month-over-month deltas, prefer LAG(...) OVER (PARTITION BY ... ORDER BY month) or add_months(..., -1) join.
    - Keep ONE final SELECT (CTEs allowed).
    """

    messages.append({"role": "system", "content": correction.strip()})

    client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = max_tokens
    )
    sql = resp.choices[0].message.content.strip()

    # strip accidental fences
    sql = re.sub(r"^\s*```(?:sql)?\s*|\s*```\s*$", "", sql, flags = re.IGNORECASE | re.DOTALL).strip()
    return sql