"""
Veridian Corp - Internal IT Service Agent
Streamlit demo UI.

Run with:
    python -m streamlit run app.py
"""

import os

import streamlit as st
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------------------------
# Streamlit configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Veridian IT Service Agent",
    page_icon="🛠️",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Database imports
# ---------------------------------------------------------------------------

from database.db import (
    init_db,
    create_request,
    get_all_tickets,
    get_all_audit_logs,
    get_ticket_queue
)

from llm.client import LLMConfigError


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

init_db()


# ---------------------------------------------------------------------------
# Page Header
# ---------------------------------------------------------------------------

st.title("🛠️ Veridian Corp — Internal IT Service Agent")

st.caption(
    "Prototype agent: retrieves policy + ticket context, generates a grounded "
    "response, checks it for hallucinations, and decides whether to resolve or escalate."
)


# ---------------------------------------------------------------------------
# API Key Check
# ---------------------------------------------------------------------------

if not os.getenv("GROQ_API_KEY"):

    st.error(
        "GROQ_API_KEY is not set. Create a `.env` file (see `.env.example`) "
        "with your Groq API key before processing a request."
    )


# ---------------------------------------------------------------------------
# Employee Request
# ---------------------------------------------------------------------------

st.header("Employee Request")


employee_name = st.text_input(
    "Employee name",
    placeholder="Enter your full name"
)


employee_id = st.text_input(
    "Employee ID",
    placeholder="Enter your employee ID"
)


employee_email = st.text_input(
    "Employee email",
    placeholder="Enter your company email"
)


request_text = st.text_area(
    "Issue / Request",
    placeholder="Describe your IT issue or request...",
    height=120
)


process_clicked = st.button(
    "Process Request",
    type="primary"
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------

if process_clicked:

    # -----------------------------------------------------------------------
    # Validate employee input
    # -----------------------------------------------------------------------

    if (
        not employee_name.strip()
        or not employee_id.strip()
        or not employee_email.strip()
        or not request_text.strip()
    ):

        st.warning(
            "Please enter your name, employee ID, email, and issue/request."
        )

    # -----------------------------------------------------------------------
    # Validate Groq configuration
    # -----------------------------------------------------------------------

    elif not os.getenv("GROQ_API_KEY"):

        st.error(
            "Cannot process request: GROQ_API_KEY is not configured."
        )

    # -----------------------------------------------------------------------
    # Process request
    # -----------------------------------------------------------------------

    else:

        with st.spinner("Running agent workflow..."):

            try:

                from graph.workflow import run_workflow

                # -----------------------------------------------------------
                # Step 1: Create employee request
                #
                # The request ID is generated automatically by the database.
                # Example:
                # REQ-001
                # REQ-002
                # REQ-003
                # -----------------------------------------------------------

                request_id = create_request(
                    employee_id=employee_id.strip(),
                    employee_name=employee_name.strip(),
                    employee_email=employee_email.strip(),
                    issue=request_text.strip()
                )

                # -----------------------------------------------------------
                # Step 2: Run LangGraph workflow
                # -----------------------------------------------------------

                result = run_workflow(
                    request_id=request_id,
                    employee_name=employee_name.strip(),
                    employee_id=employee_id.strip(),
                    employee_email=employee_email.strip(),
                    original_request=request_text.strip()
                )

                # -----------------------------------------------------------
                # Step 3: Store result in Streamlit session
                # -----------------------------------------------------------

                st.session_state["result"] = result

            # ---------------------------------------------------------------
            # LLM configuration error
            # ---------------------------------------------------------------

            except LLMConfigError as e:

                st.error(str(e))

            # ---------------------------------------------------------------
            # Unexpected error
            # ---------------------------------------------------------------

            except Exception as e:

                st.error(
                    "Something went wrong while processing the request. "
                    "Please try again."
                )

                st.exception(e)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()


    # -----------------------------------------------------------------------
    # Agent Analysis
    # -----------------------------------------------------------------------

    with st.container(border=True):

        st.header("Agent Analysis")

        # Automatically generated request ID
        st.caption(
            f"Request ID: {result.get('request_id', '-')}"
        )

        # Intent
        st.markdown("**Intent**")

        st.write(
            result.get("intent", "-")
        )

        # Category + Extracted Details
        a1, a2 = st.columns(2)

        with a1:

            st.markdown("**Category**")

            st.write(
                result.get("category", "-")
            )

        


    # -----------------------------------------------------------------------
    # Agent Response
    # -----------------------------------------------------------------------

    with st.container(border=True):

        st.header("Agent Response")

        st.write(
            result.get("generated_response", "")
        )


    # -----------------------------------------------------------------------
    # Outcome
    # -----------------------------------------------------------------------

    with st.container(border=True):

        st.header("Outcome")

        decision = (
            result.get("decision")
            or result.get("action")
        )

        # ---------------------------------------------------------------
        # RESOLVE
        # ---------------------------------------------------------------

        if decision == "RESOLVE":

            st.success("RESOLVE")

            st.caption(
                "Grounding check passed • No ticket created"
            )

        # ---------------------------------------------------------------
        # ESCALATE
        # ---------------------------------------------------------------

        else:

            ticket_id = result.get("ticket_id")

            st.warning("⚠ ESCALATED")

            if ticket_id:

                st.caption(
                    f"Grounding check "
                    f"{'passed' if result.get('grounded') else 'failed'} "
                    f"• Ticket: {ticket_id}"
                )

            else:

                st.caption(
                    f"Grounding check "
                    f"{'passed' if result.get('grounded') else 'failed'}"
                )


    # -----------------------------------------------------------------------
    # Audit Trail
    # -----------------------------------------------------------------------

    with st.expander("▶ View Audit Trail"):

        audit = result.get(
            "audit_data",
            {}
        )

        st.json(audit)


# ---------------------------------------------------------------------------
# Sidebar: Reference Data
# ---------------------------------------------------------------------------

with st.sidebar:

    st.header("Reference Data")


    # -----------------------------------------------------------------------
    # Existing Data Pack Ticket Queue
    # -----------------------------------------------------------------------

    with st.expander("📋 Existing Ticket Queue (Data Pack)"):

        st.dataframe(
            get_ticket_queue(),
            use_container_width=True,
            hide_index=True
        )


    # -----------------------------------------------------------------------
    # Agent-created Tickets
    # -----------------------------------------------------------------------

    with st.expander("🎫 Tickets Created By Agent"):

        st.dataframe(
            get_all_tickets(),
            use_container_width=True,
            hide_index=True
        )


    # -----------------------------------------------------------------------
    # Audit Logs
    # -----------------------------------------------------------------------

    with st.expander("📝 Audit Log"):

        st.dataframe(
            get_all_audit_logs(),
            use_container_width=True,
            hide_index=True
        )


    st.divider()