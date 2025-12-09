# orchestration/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, END

from orchestration.state import TelecomAssistantState
from agents.billing_agents import process_billing_query
from agents.network_agents import process_network_query
from agents.service_agents import recommend_personalized_plan
from agents.knowledge_agents import answer_knowledge_query


# ---------- Classification & Routing ----------

def classify_query(state: TelecomAssistantState) -> TelecomAssistantState:
    """Very simple keyword-based classifier (can be upgraded to LLM later)."""
    q = state["query"].lower()
    cls = "fallback"

    if any(w in q for w in ["bill", "charge", "payment", "account", "due date", "invoice"]):
        cls = "billing_account"
    elif any(w in q for w in ["network", "signal", "connection", "call", "data", "slow", "internet", "5g", "4g"]):
        cls = "network_troubleshooting"
    elif any(w in q for w in ["plan", "recommend", "upgrade", "downgrade", "best", "family", "pack"]):
        cls = "service_recommendation"
    elif any(w in q for w in ["how", "what", "configure", "setup", "apn", "volte", "roaming", "esim"]):
        cls = "knowledge_retrieval"

    return {**state, "classification": cls}


def route_query(state: TelecomAssistantState) -> str:
    """Choose which backend node to call based on classification."""
    cls = state.get("classification", "fallback")
    if cls == "billing_account":
        return "crew_ai_node"
    if cls == "network_troubleshooting":
        return "autogen_node"
    if cls == "service_recommendation":
        return "langchain_node"
    if cls == "knowledge_retrieval":
        return "llamaindex_node"
    return "fallback_handler"


# ---------- Backend Nodes ----------

def crew_ai_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle billing & account queries using CrewAI + telecom.db."""
    query = state["query"]
    customer_info = state.get("customer_info") or {}
    customer_identifier = customer_info.get("email") or customer_info.get("id")

    try:
        answer = process_billing_query(customer_identifier=customer_identifier, query=query)
    except Exception as e:
        answer = (
            "Sorry, I ran into a problem while analyzing your billing question.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )

    return {
        **state,
        "intermediate_responses": {"crew_ai": answer},
    }


def autogen_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle network troubleshooting queries using our 2-agent network workflow."""
    query = state["query"]
    customer_info = state.get("customer_info") or {}
    customer_identifier = customer_info.get("email") or customer_info.get("id")

    try:
        answer = process_network_query(query, customer_identifier)
    except Exception as e:
        answer = (
            "Sorry, I ran into a problem while diagnosing your network issue.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )

    return {
        **state,
        "intermediate_responses": {"autogen": answer},
    }


def langchain_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle plan recommendations using LangChain + real DB data."""
    query = state["query"]
    customer_info = state.get("customer_info") or {}
    customer_email = customer_info.get("email")

    try:
        answer = recommend_personalized_plan(customer_email, query)
    except Exception as e:
        answer = (
            "Sorry, I had trouble generating a plan recommendation.\n\n"
            f"Technical details: {type(e).__name__}: {e}"
        )

    return {
        **state,
        "intermediate_responses": {"langchain": answer},
    }


def llamaindex_node(state: TelecomAssistantState) -> TelecomAssistantState:
    """Handle knowledge / how-to questions using LlamaIndex over docs."""
    query = state["query"]
    customer_info = state.get("customer_info") or {}
    customer_email = customer_info.get("email")

    answer = answer_knowledge_query(query, customer_email)

    return {
        **state,
        "intermediate_responses": {"llamaindex": answer},
    }


def fallback_handler(state: TelecomAssistantState) -> TelecomAssistantState:
    """Fallback when classification is unclear."""
    txt = (
        "I’m not sure how to handle that.\n\n"
        "You can ask about:\n"
        "- Billing & account (charges, due dates, invoices)\n"
        "- Network issues (slow internet, no signal, call drops)\n"
        "- Plan recommendations (upgrade/downgrade, family plans)\n"
        "- Technical how-to (APN settings, VoLTE, roaming, eSIM)"
    )
    return {**state, "intermediate_responses": {"fallback": txt}}


# ---------- Response Aggregation ----------

def formulate_response(state: TelecomAssistantState) -> TelecomAssistantState:
    """Pick the backend's answer and store it as final_response."""
    responses = state.get("intermediate_responses", {})
    if responses:
        # just pick the first backend response for now
        final = next(iter(responses.values()))
    else:
        final = "Something went wrong while generating a response."
    return {**state, "final_response": final}


# ---------- Graph Factory ----------

def create_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(TelecomAssistantState)

    # Nodes
    workflow.add_node("classify_query", classify_query)
    workflow.add_node("crew_ai_node", crew_ai_node)
    workflow.add_node("autogen_node", autogen_node)
    workflow.add_node("langchain_node", langchain_node)
    workflow.add_node("llamaindex_node", llamaindex_node)
    workflow.add_node("fallback_handler", fallback_handler)
    workflow.add_node("formulate_response", formulate_response)

    # Routing after classification
    workflow.add_conditional_edges(
        "classify_query",
        route_query,
        {
            "crew_ai_node": "crew_ai_node",
            "autogen_node": "autogen_node",
            "langchain_node": "langchain_node",
            "llamaindex_node": "llamaindex_node",
            "fallback_handler": "fallback_handler",
        },
    )

    # After backend nodes → formulate_response → END
    workflow.add_edge("crew_ai_node", "formulate_response")
    workflow.add_edge("autogen_node", "formulate_response")
    workflow.add_edge("langchain_node", "formulate_response")
    workflow.add_edge("llamaindex_node", "formulate_response")
    workflow.add_edge("fallback_handler", "formulate_response")
    workflow.add_edge("formulate_response", END)

    # Entry point
    workflow.set_entry_point("classify_query")

    return workflow.compile()
