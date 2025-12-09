# agents/service_agents.py
from __future__ import annotations
 
from typing import Optional, Dict, Any, List
 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
 
from utils.database import run_query, get_tables
from config.config import get_openai_api_key, get_openai_model
 
 
# ---------------------------------------------------------
# Helpers to load customer profile
# ---------------------------------------------------------
 
def _load_customer_profile(customer_email: Optional[str]) -> Dict[str, Any]:
    """
    Load customer record, current plan, last billing-period usage,
    and all available plans from the DB.
    """
 
    profile: Dict[str, Any] = {
        "customer": None,
        "current_plan": None,
        "usage": None,
        "plans": None,
    }
 
    # Load all plans
    tables = get_tables()
    if "service_plans" in tables:
        profile["plans"] = run_query("SELECT * FROM service_plans")
 
    if not customer_email:
        return profile
 
    # Load customer details
    rows = run_query(
        "SELECT * FROM customers WHERE email = :email LIMIT 1",
        {"email": customer_email},
    )
    if not rows:
        return profile
 
    customer = rows[0]
    profile["customer"] = customer
 
    # Load current plan
    if customer.get("service_plan_id") and profile["plans"]:
        plan_id = customer["service_plan_id"]
        for p in profile["plans"]:
            if p["plan_id"] == plan_id:
                profile["current_plan"] = p
                break
 
    # Load latest usage (correct columns + ordering)
    if "customer_usage" in tables:
        usage_rows = run_query(
            """
            SELECT * FROM customer_usage
            WHERE customer_id = :cid
            ORDER BY billing_period_end DESC
            LIMIT 1
            """,
            {"cid": customer["customer_id"]},
        )
        if usage_rows:
            profile["usage"] = usage_rows[0]
 
    return profile
 
 
# ---------------------------------------------------------
# Format plan for display
# ---------------------------------------------------------
 
def _format_plan(p: dict) -> str:
    data = "Unlimited" if p["unlimited_data"] else f"{p['data_limit_gb']}GB"
    voice = "Unlimited" if p["unlimited_voice"] else f"{p['voice_minutes']} min"
    sms = "Unlimited" if p["unlimited_sms"] else f"{p['sms_count']} SMS"
 
    return (
        f"- {p['name']} ({p['plan_id']}): ₹{p['monthly_cost']} / month | "
        f"{data} | {voice} | {sms}"
    )
 
 
# ---------------------------------------------------------
# LangChain LLM builder
# ---------------------------------------------------------
 
def _build_llm() -> ChatOpenAI:
    key = get_openai_api_key()
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")
 
    return ChatOpenAI(
        model=get_openai_model(),
        temperature=0.2,
        openai_api_key=key,
    )
 
 
# ---------------------------------------------------------
# Personalized Plan Recommendation (Version C)
# ---------------------------------------------------------
 
def recommend_personalized_plan(customer_email: Optional[str], user_query: str) -> str:
    """
    Personalized plan suggestion using:
    - customer profile
    - real usage (from customer_usage)
    - current plan features
    - all available service_plans
    """
 
    profile = _load_customer_profile(customer_email)
 
    plans: List[dict] = profile.get("plans", []) or []
    current_plan = profile.get("current_plan")
    usage = profile.get("usage")
    customer = profile.get("customer")
 
    # Format plans
    plan_text = "\n".join(_format_plan(p) for p in plans) if plans else "(No plans)"
 
    # Format customer profile
    if customer and current_plan:
        customer_text = (
            f"Customer ID: {customer['customer_id']}\n"
            f"Name: {customer['name']}\n"
            f"Current Plan: {current_plan['name']} ({current_plan['plan_id']})\n"
            f"Monthly Cost: ₹{current_plan['monthly_cost']}\n"
        )
    else:
        customer_text = "No customer profile available.\n"
 
    # Format usage (using correct column names)
    if usage:
        customer_usage_text = (
            f"Billing Period: {usage['billing_period_start']} → {usage['billing_period_end']}\n"
            f"Last Billing Period Usage:\n"
            f"- Data: {usage['data_used_gb']} GB\n"
            f"- Voice: {usage['voice_minutes_used']} min\n"
            f"- SMS: {usage['sms_count_used']}\n"
            f"- Total Bill Amount: ₹{usage['total_bill_amount']}\n"
        )
    else:
        customer_usage_text = "No usage data available.\n"
 
    llm = _build_llm()
 
    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Telecom Plan Optimization Expert for an Indian telecom operator.
 
You must recommend 1–3 plans based on:
1. User's request in natural language (e.g., cheaper plan, more data, family plan, add-on packs).
2. Current plan vs actual usage (data, voice, SMS).
3. Cost savings or upgrade benefits.
4. All available plans from `service_plans`.
5. Unlimited data/voice/SMS rules.
6. Whether the current plan is already ideal.
 
Also consider add-on packs or upgrades if the user is exceeding data/voice/SMS limits.
If the question is specifically about "add-on packs", clearly list suitable add-on options
based on their current plan and usage, and explain when they should upgrade the base plan instead.
 
Respond with:
- A short summary
- A ranked list of recommended plans and/or add-on packs
- Reasoning in bullet points
                """,
            ),
            (
                "user",
                """
User Query:
{query}
 
Customer:
{customer_profile}
 
Usage:
{usage}
 
Available Plans:
{plans}
 
Provide the best personalized plan/add-on recommendations.
                """,
            ),
        ]
    )
 
    messages = prompt.format_messages(
        query=user_query,
        customer_profile=customer_text,
        usage=customer_usage_text,
        plans=plan_text,
    )
 
    response = llm.invoke(messages)
    return response.content
 
 