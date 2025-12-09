# orchestration/state.py
from __future__ import annotations

from typing import TypedDict, Dict, Any, List


class TelecomAssistantState(TypedDict):
    """
    Shared state structure for the LangGraph workflow.

    Fields:
      - query: The user's latest query/message.
      - customer_info: Info from login (email, id, etc.).
      - classification: High-level type of query (billing, network, etc.).
      - intermediate_responses: Raw responses from backend agents.
      - final_response: The message that will be shown in the UI.
      - chat_history: Simple list of previous user/assistant messages.
    """
    query: str
    customer_info: Dict[str, Any]
    classification: str
    intermediate_responses: Dict[str, Any]
    final_response: str
    chat_history: List[Dict[str, str]]
