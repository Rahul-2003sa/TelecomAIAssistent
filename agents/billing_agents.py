# agents/billing_agents.py
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from crewai import Agent, Task, Crew, LLM

from utils.database import get_db_uri, get_tables, run_query
from config.config import get_openai_api_key, get_openai_model


def _build_llm() -> LLM:
    """
    Create a CrewAI LLM instance using OpenAI.

    Requires OPENAI_API_KEY in environment (loaded via dotenv in config).
    """
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Please add it to your .env file."
        )

    # CrewAI also reads OPENAI_API_KEY from env
    os.environ["OPENAI_API_KEY"] = api_key

    return LLM(
        model=get_openai_model(),  # e.g. "gpt-4o-mini"
        temperature=0.2,
    )


def _format_rows(rows: List[Dict[str, Any]], max_rows: int = 5) -> str:
    """Turn a list of dict rows into a readable text table snippet."""
    if not rows:
        return "  (no rows)\n"

    rows = rows[:max_rows]
    columns = list(rows[0].keys())

    lines = []
    lines.append("  Columns: " + ", ".join(columns))
    for r in rows:
        parts = [f"{col}={r[col]!r}" for col in columns]
        lines.append("  - " + ", ".join(parts))
    return "\n".join(lines) + "\n"


def _build_db_snapshot(customer_identifier: Optional[str]) -> str:
    """
    Build a textual snapshot of relevant DB data for the customer.

    Strategy:
      - List all tables.
      - For each table, show up to a few rows.
      - If we have a customer identifier, try some common column names to filter.
      - Ignore any SQL errors (we don't want to crash the app).
    """
    lines: List[str] = []

    lines.append(f"Database URI: {get_db_uri()}")
    lines.append("Discovered tables:")

    try:
        tables = get_tables()
    except Exception as e:
        return f"(Failed to inspect database tables: {type(e).__name__}: {e})"

    if not tables:
        return "(No tables found in the database.)"

    for table in tables:
        lines.append(f"\nTable: {table}")

        # First, try to filter by common customer identifier columns
        filtered_rows: List[Dict[str, Any]] = []
        if customer_identifier:
            candidate_columns = ["customer_id", "customerid", "email", "phone", "msisdn", "account_id"]
            for col in candidate_columns:
                try:
                    filtered_rows = run_query(
                        f"SELECT * FROM {table} WHERE {col} = :val LIMIT 5",
                        {"val": customer_identifier},
                    )
                    if filtered_rows:
                        lines.append(f"  Rows for {col} = {customer_identifier!r}:")
                        lines.append(_format_rows(filtered_rows))
                        break  # stop trying other columns for this table
                except Exception:
                    # Column might not exist in this table; just ignore
                    continue

        # If we didn't find any filtered rows, just show a sample of the table
        if not filtered_rows:
            try:
                sample_rows = run_query(f"SELECT * FROM {table} LIMIT 5")
                lines.append("  Sample rows:")
                lines.append(_format_rows(sample_rows))
            except Exception as e:
                lines.append(f"  (Failed to query table {table}: {type(e).__name__}: {e})")

    return "\n".join(lines)


def create_billing_crew(db_snapshot: str) -> Crew:
    """
    Create a CrewAI crew for telecom billing & account queries.

    Agents:
      - Billing Specialist: uses the DB snapshot text to investigate
      - Service Advisor: explains results clearly to customer
    """
    llm = _build_llm()

    billing_specialist = Agent(
        role="Telecom Billing Specialist",
        goal=(
            "Investigate and explain customer billing and account questions using "
            "the provided database snapshot of the telecom billing system."
        ),
        backstory=(
            "You are an expert telecom billing analyst. You understand billing cycles, "
            "taxes, discounts, data/voice/SMS usage and pro-rating rules. "
            "You are given a textual snapshot of database tables and sample rows. "
            "Use that information carefully to infer what is happening with the "
            "customer's charges."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    service_advisor = Agent(
        role="Customer Service Advisor",
        goal=(
            "Turn the technical billing investigation into a clear, friendly "
            "explanation with next steps for the customer."
        ),
        backstory=(
            "You work in customer care. You explain complex billing issues in simple "
            "language and suggest helpful actions like plan changes or usage tips."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    billing_task = Task(
        description=(
            "You are given:\n"
            "1) The customer's identifier (email / phone / id).\n"
            "2) The customer's billing question.\n"
            "3) A snapshot of relevant database tables and rows from the telecom "
            "billing system.\n\n"
            "Use ONLY the information in the database snapshot and logical reasoning "
            "to investigate the most likely reasons for the charges the customer is "
            "asking about.\n\n"
            "Customer identifier: {customer_identifier}\n"
            "Customer question: {query}\n\n"
            "Database snapshot:\n"
            "{db_snapshot}\n"
        ),
        expected_output=(
            "A concise technical summary of what you found in the database snapshot: "
            "which tables/rows are relevant and what they show about the customer's "
            "bills and charges."
        ),
        agent=billing_specialist,
    )

    summary_task = Task(
        description=(
            "Using the technical summary from the billing specialist, write a "
            "clear, friendly explanation for the customer.\n\n"
            "Make sure to:\n"
            "- Answer their question directly.\n"
            "- Explain the main reasons for their current charges.\n"
            "- Mention any one-time fees, extra usage, or discounts if present.\n"
            "- Suggest at most one or two plan changes or tips if helpful.\n"
        ),
        expected_output=(
            "A customer-facing explanation in 3–6 short paragraphs, plus bullet "
            "points for key amounts if appropriate."
        ),
        agent=service_advisor,
    )

    crew = Crew(
        agents=[billing_specialist, service_advisor],
        tasks=[billing_task, summary_task],
        verbose=True,
    )
    return crew


def process_billing_query(
    customer_identifier: Optional[str],
    query: str,
) -> str:
    """
    High-level function used by the LangGraph node.

    customer_identifier: email / phone / customer_id (whatever you use to look up the user)
    query: natural language billing question
    """
    # Build a textual snapshot of DB content for this customer
    db_snapshot = _build_db_snapshot(customer_identifier)

    crew = create_billing_crew(db_snapshot=db_snapshot)

    result = crew.kickoff(
        inputs={
            "customer_identifier": customer_identifier or "unknown",
            "query": query,
            "db_snapshot": db_snapshot,
        }
    )

    return str(result)
