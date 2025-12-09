# agents/network_agents.py
from __future__ import annotations

from typing import Optional
from openai import OpenAI

from config.config import get_openai_api_key, get_openai_model


def _build_client() -> OpenAI:
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in your environment or .env")
    return OpenAI(api_key=key)


def _diagnostics_agent(query: str, customer_id: Optional[str]) -> str:
    client = _build_client()

    system_prompt = """
You are a Telecom Network Diagnostics Specialist.
Analyze network issues including:
- slow internet
- no signal
- call drops
- VoLTE not working
- 5G issues
- SIM/network registration failures

Follow this structure:
1. Identify possible root causes
2. Check if it is device-related or network-area-related
3. Mention what parameters need to be checked (signal strength, APN, VoLTE toggle)
4. Give a technical summary (NOT customer-facing)
"""

    message = (
        f"Customer identifier: {customer_id}\n"
        f"Reported issue: {query}\n"
        "Generate your technical diagnostic analysis:"
    )

    response = client.chat.completions.create(
        model=get_openai_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


def _resolution_agent(diagnostics_summary: str) -> str:
    client = _build_client()

    system_prompt = """
You are a Telecom Network Resolution Specialist.
Your job is to convert technical diagnostics into:
- a clear explanation
- friendly language
- 3–5 clear steps the customer can follow
- a possible root cause
- optional tips to prevent it in future

Do NOT be too technical. Be helpful.
"""

    user_msg = (
        "Here is the technical diagnostics summary from the engineering system:\n\n"
        f"{diagnostics_summary}\n\n"
        "Convert this into a helpful customer-facing explanation with steps."
    )

    response = client.chat.completions.create(
        model=get_openai_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content


def process_network_query(query: str, customer_identifier: Optional[str] = None) -> str:
    """
    High-level method used by LangGraph node.
    Multi-agent flow:
        Diagnostics Agent  → Resolution Agent
    """

    try:
        diagnostics = _diagnostics_agent(query, customer_identifier)
        resolution = _resolution_agent(diagnostics)

        final_response = (
            "### Network Diagnostics (Internal)\n"
            f"{diagnostics}\n\n"
            "### Final Explanation\n"
            f"{resolution}"
        )
        return final_response

    except Exception as e:
        return f"Network troubleshooting failed: {type(e).__name__}: {e}"
