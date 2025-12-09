# ui/streamlit_app.py

import sys
from pathlib import Path
from typing import TypedDict, List, Dict, Any

import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------
# 1. Make sure we can import from project root (.. = parent of ui/)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now these imports should work as long as your project structure is:
# Project/
#   orchestration/
#   ui/
from orchestration.graph import create_graph, TelecomAssistantState


# ---------------------------------------------------------------------
# 2. Session initialization
# ---------------------------------------------------------------------
def init_session() -> None:
    """Initialize keys in Streamlit session state."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_type" not in st.session_state:
        st.session_state.user_type = None
    if "email" not in st.session_state:
        st.session_state.email = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Dict[str, str]] = []

    # Initialize graph only once
    if "graph" not in st.session_state:
        try:
            st.session_state.graph = create_graph()
            st.session_state.graph_loaded = True
            st.session_state.graph_error = None
        except Exception as e:
            st.session_state.graph_loaded = False
            st.session_state.graph_error = str(e)
            st.session_state.graph = None


# ---------------------------------------------------------------------
# 3. Helper to run LangGraph flow
# ---------------------------------------------------------------------
def run_query_through_graph(query: str) -> str:
    """Send a query through our LangGraph workflow and return the final response."""

    # If graph failed to initialize, don't crash – show a friendly message
    if not st.session_state.get("graph_loaded", False) or st.session_state.graph is None:
        error_msg = st.session_state.get(
            "graph_error",
            "Graph is not initialized. Please check backend / database configuration.",
        )
        return f"⚠️ System initialization error: {error_msg}"

    # Prepare state for LangGraph
    state: TelecomAssistantState = {
        "query": query,
        "customer_info": {"email": st.session_state.email},
        "classification": "",
        "intermediate_responses": {},
        "final_response": "",
        "chat_history": st.session_state.chat_history,
    }

    try:
        result: TelecomAssistantState = st.session_state.graph.invoke(state)
    except Exception as e:
        # If your database / agents fail, you'll see the error here
        return f"❌ Error while processing your request (possible DB/agent issue): {e}"

    # Fall back if the key is missing
    return result.get("final_response", "I couldn't generate a response.")


# ---------------------------------------------------------------------
# 4. Sidebar login / logout
# ---------------------------------------------------------------------
def sidebar_login() -> None:
    with st.sidebar:
        st.title("Telecom Service Assistant")

        # If graph creation failed, show it in sidebar for debugging
        if st.session_state.get("graph_error"):
            st.error(
                "Graph initialization failed.\n\n"
                f"Details: {st.session_state.graph_error}"
            )

        if not st.session_state.authenticated:
            st.subheader("Login")
            email = st.text_input("Email Address")
            user_type = st.selectbox("User Type", ["Customer", "Admin"])

            if st.button("Login"):
                if email and "@" in email:
                    st.session_state.authenticated = True
                    st.session_state.user_type = user_type
                    st.session_state.email = email
                    st.success(f"Logged in as {user_type}")
                    st.rerun()  # reload UI with authenticated state
                else:
                    st.error("Please enter a valid email address")
        else:
            st.success(f"Logged in as {st.session_state.user_type}")
            st.text(f"Email: {st.session_state.email}")
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.user_type = None
                st.session_state.email = None
                st.session_state.chat_history = []
                st.rerun()


# ---------------------------------------------------------------------
# 5. Customer view
# ---------------------------------------------------------------------
def customer_view() -> None:
    st.title("Welcome to Telecom Service Assistant")

    tab1, tab2, tab3 = st.tabs(["Chat Assistant", "My Account", "Network Status"])

    # ---- Chat tab ----
    with tab1:
        st.header("Chat with our AI Assistant")

        # Show previous messages
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # User input
        if prompt := st.chat_input("How can I help you today?"):
            # Store user message
            st.session_state.chat_history.append(
                {"role": "user", "content": prompt}
            )
            with st.chat_message("user"):
                st.write(prompt)

            # Model response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = run_query_through_graph(prompt)
                    st.write(response)

            # Add assistant response to history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

    # ---- My Account tab (demo / can later connect to DB) ----
    with tab2:
        st.header("My Account Information")
        st.subheader("Current Plan")
        st.write("Standard Plan (STD_500)")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Data Used", "3.5 GB", "2.1 GB remaining")
        with col2:
            st.metric("Voice Minutes", "320 mins", "Unlimited")
        with col3:
            st.metric("SMS Used", "45", "Unlimited")

        st.subheader("Billing Information")
        st.write("Next Bill Date: June 15, 2023")
        st.write("Monthly Charge: ₹799.00")

    # ---- Network Status tab (demo data) ----
    with tab3:
        st.header("Network Status")
        status_df = pd.DataFrame(
            {
                "Region": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"],
                "4G Status": ["Normal", "Normal", "Degraded", "Normal", "Normal"],
                "5G Status": ["Normal", "Maintenance", "Normal", "Normal", "Degraded"],
            }
        )
        st.dataframe(status_df, use_container_width=True)
        st.subheader("Known Issues")
        st.info("Scheduled maintenance in Delhi region (03:00–05:00 AM)")
        st.warning("Network congestion reported in Bangalore South")


# ---------------------------------------------------------------------
# 6. Admin view
# ---------------------------------------------------------------------
def admin_view() -> None:
    st.title("Admin Dashboard")

    tab1, tab2, tab3 = st.tabs(
        ["Knowledge Base Management", "Customer Support", "Network Monitoring"]
    )

    # ---- Knowledge base tab ----
    with tab1:
        st.header("Knowledge Base Management")
        st.subheader("Upload Documents to Knowledge Base")

        uploaded_files = st.file_uploader(
            "Upload PDF, Markdown, or Text files",
            type=["pdf", "md", "txt"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            for f in uploaded_files:
                # TODO: Wire this to your LlamaIndex / vector DB ingestion
                st.success(
                    f"Processed {f.name} and added to knowledge base (demo only)."
                )

        st.subheader("Existing Documents")
        doc_df = pd.DataFrame(
            {
                "Document Name": [
                    "Service Plans Guide.md",
                    "Network Troubleshooting Guide.md",
                    "Billing FAQs.md",
                    "Technical Support Guide.md",
                ],
                "Type": ["Markdown"] * 4,
                "Last Updated": ["2023-06-20", "2023-06-18", "2023-06-15", "2023-06-10"],
            }
        )
        st.dataframe(doc_df, use_container_width=True)

    # ---- Support tab ----
    with tab2:
        st.header("Customer Support Dashboard")
        ticket_df = pd.DataFrame(
            {
                "Ticket ID": ["TKT004", "TKT005"],
                "Customer": ["Ananya Singh", "Vikram Reddy"],
                "Issue": ["Account reactivation", "Slow internet speeds"],
                "Status": ["In Progress", "Assigned"],
                "Priority": ["Medium", "Medium"],
                "Created": ["2023-06-15", "2023-06-17"],
            }
        )
        st.dataframe(ticket_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Open Tickets", "2", "-3")
        with col2:
            st.metric("Avg. Resolution Time", "4.3 hours", "-0.5")
        with col3:
            st.metric("Customer Satisfaction", "92%", "+3%")

    # ---- Network monitoring tab ----
    with tab3:
        st.header("Network Monitoring")
        incident_df = pd.DataFrame(
            {
                "Incident ID": ["INC003"],
                "Type": ["Equipment Failure"],
                "Location": ["Delhi West"],
                "Affected Services": ["Voice, Data, SMS"],
                "Started": ["2023-06-15 08:15:00"],
                "Status": ["In Progress"],
                "Severity": ["Critical"],
            }
        )
        st.dataframe(incident_df, use_container_width=True)


# ---------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Telecom Service Assistant",
        page_icon="📶",
        layout="wide",
    )

    init_session()
    sidebar_login()

    if st.session_state.authenticated:
        if st.session_state.user_type == "Customer":
            customer_view()
        else:
            admin_view()
    else:
        st.title("Telecom Service Assistant")
        st.write("Please log in from the sidebar to continue.")


if __name__ == "__main__":
    main()
