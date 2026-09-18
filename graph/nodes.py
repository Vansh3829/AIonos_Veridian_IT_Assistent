"""
LangGraph node functions.

Each function takes the current AgentState dict and returns a dict of the
fields it updates. LangGraph merges these into the running state.

Flow:

    analyze_request -> retrieve_context -> [policy_found?]
        NO  -> create_ticket -> audit_log -> END
        YES -> generate_response -> check_grounding -> [grounded?]
                NO  -> create_ticket -> audit_log -> END
                YES -> decide_action -> [decision?]
                        RESOLVE  -> audit_log -> END
                        ESCALATE -> create_ticket -> audit_log -> END
"""

from llm.client import call_llm, call_llm_json

from rag.retriever import (
    retrieve_policies,
    has_relevant_policy,
    relevant_policies_only,
    retrieve_tickets,
)

from database.db import create_ticket, log_audit


CATEGORIES = [
    "Password",
    "VPN",
    "Laptop",
    "Software",
    "Printer",
    "Mailbox",
    "Guest Wi-Fi",
    "Expense Software",
    "Security Incident",
    "WFH Equipment",
    "Other",
]


NO_POLICY_MESSAGE = (
    "No relevant Veridian Corp policy was found for this request. "
    "The request requires human review."
)


GROUNDING_FAIL_MESSAGE = (
    "The agent could not produce a sufficiently grounded answer based on "
    "Veridian Corp's documented policies for this request. "
    "It has been escalated for human review."
)


# ---------------------------------------------------------------------------
# 1. analyze_request
# ---------------------------------------------------------------------------

def analyze_request(state):
    """Classify the employee's request and extract relevant details."""

    request_text = state["original_request"]

    system_prompt = (
        "You are an intent classifier for Veridian Corp's internal IT support "
        "system. Classify the employee's request. Respond with ONLY a JSON "
        "object, no other text.\n\n"

        f"Allowed categories (pick exactly one): "
        f"{', '.join(CATEGORIES)}\n\n"

        "Extract the employee's intent, category, important details, and "
        "keywords that can help retrieve the relevant IT policy and existing "
        "tickets.\n\n"

        "Do not invent information that is not present in the employee's "
        "request.\n\n"

        "JSON format:\n"
        "{\n"
        '  "intent": "<short phrase describing what the employee wants>",\n'
        '  "category": "<one of the allowed categories>",\n'
        '  "extracted_details": "<key facts/entities mentioned>",\n'
        '  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"]\n'
        "}"
    )

    fallback = {
        "intent": "Unclear",
        "category": "Other",
        "extracted_details": "",
        "keywords": request_text.split()[:5],
    }

    result = call_llm_json(
        system_prompt,
        f"Employee request: {request_text}",
        fallback,
    )

    category = result.get("category", "Other")

    if category not in CATEGORIES:
        category = "Other"

    return {
        "intent": result.get("intent", "Unclear"),
        "category": category,
        "extracted_details": result.get("extracted_details", ""),
        "keywords": result.get("keywords", []) or [category],
    }


# ---------------------------------------------------------------------------
# 2. retrieve_context
# ---------------------------------------------------------------------------

def retrieve_context(state):
    """Retrieve relevant policy chunks and existing tickets."""

    query = (
        f"{state.get('intent', '')} "
        f"{state.get('category', '')} "
        f"{state['original_request']}"
    )

    keywords = state.get("keywords", []) + [
        state.get("category", "")
    ]

    policies = retrieve_policies(query)
    tickets = retrieve_tickets(keywords)

    policy_found = has_relevant_policy(policies)

    return {
        "retrieved_policies": (
            relevant_policies_only(policies)
            if policy_found
            else policies
        ),
        "retrieved_tickets": tickets,
        "policy_found": policy_found,
    }


def route_after_retrieve(state):
    """Conditional edge: does a relevant policy exist?"""

    return (
        "generate_response"
        if state.get("policy_found")
        else "no_policy"
    )


# ---------------------------------------------------------------------------
# 3. generate_response
# ---------------------------------------------------------------------------

def generate_response(state):
    """Generate a grounded response using only policy and ticket context."""

    policy_context = "\n\n".join(
        f"[{p['source']}]\n{p['content']}"
        for p in state.get("retrieved_policies", [])
    )

    ticket_context = "\n".join(
        f"- {t['ticket_id']} ({t['employee']}): "
        f"{t['issue_summary']} - status: {t['status']}"
        for t in state.get("retrieved_tickets", [])
    ) or "No related existing tickets."

    system_prompt = (
        "You are Veridian Corp's internal IT support agent.\n\n"

        "Answer the employee using ONLY the supplied Veridian Corp policy "
        "context and relevant ticket context below.\n\n"

        "Do not invent:\n"
        "- policies\n"
        "- procedures\n"
        "- approval requirements\n"
        "- timelines\n"
        "- permissions\n"
        "- technical instructions\n"
        "- company rules\n\n"

        "If the information is not supported by the provided context, "
        "do not state it as fact.\n\n"

        "Give a concise, professional employee-facing response in "
        "1-3 short sentences.\n"

        "State only the action the employee should take and any directly "
        "relevant policy requirement.\n"

        "Do not include headings, bullet points, analysis, or explanations "
        "about the agent's reasoning.\n"

        "Do not repeat information unnecessarily.\n\n"

        f"--- POLICY CONTEXT ---\n{policy_context}\n\n"
        f"--- RELEVANT TICKET CONTEXT ---\n{ticket_context}\n"
    )

    user_prompt = (
        f"Employee request: {state['original_request']}"
    )

    response = call_llm(
        system_prompt,
        user_prompt,
    )

    return {
        "generated_response": response
    }


# ---------------------------------------------------------------------------
# 4. check_grounding
# ---------------------------------------------------------------------------

def check_grounding(state):
    """Verify that the generated response is supported by policy context."""

    policy_context = "\n\n".join(
        f"[{p['source']}]\n{p['content']}"
        for p in state.get("retrieved_policies", [])
    )

    system_prompt = (
        "You are a strict grounding/hallucination checker. You will be "
        "given a POLICY CONTEXT and a RESPONSE that an IT support agent "
        "generated for an employee.\n\n"

        "Determine whether every factual claim in the RESPONSE "
        "(procedures, approvals, timelines, who does what) is actually "
        "supported by the POLICY CONTEXT.\n\n"

        "If the response states anything not backed by the context, "
        "it is NOT grounded.\n\n"

        "Respond with ONLY a JSON object:\n"
        '{"grounded": true or false, '
        '"reason": "<short explanation>"}'
    )

    user_prompt = (
        f"POLICY CONTEXT:\n{policy_context}\n\n"
        f"RESPONSE TO CHECK:\n"
        f"{state.get('generated_response', '')}"
    )

    fallback = {
        "grounded": False,
        "reason": (
            "Grounding check could not be parsed; failing safe."
        ),
    }

    result = call_llm_json(
        system_prompt,
        user_prompt,
        fallback,
    )

    return {
        "grounded": bool(result.get("grounded", False)),
        "grounding_reason": result.get("reason", ""),
    }


def route_after_grounding(state):
    """Conditional edge: did the response pass grounding?"""

    return (
        "decide_action"
        if state.get("grounded")
        else "grounding_failed"
    )


# ---------------------------------------------------------------------------
# 5. decide_action
# ---------------------------------------------------------------------------

def decide_action(state):
    """Decide RESOLVE vs ESCALATE."""

    policy_context = "\n\n".join(
        f"[{p['source']}]\n{p['content']}"
        for p in state.get("retrieved_policies", [])
    )

    system_prompt = (
        "You decide whether Veridian Corp's IT team can directly RESOLVE "
        "an employee request, or whether it must be ESCALATED.\n\n"

        "ESCALATE when the request requires approval from someone else "
        "(for example, a manager, Finance, or Security review), or is not "
        "something IT alone can complete.\n\n"

        "Base your decision only on the POLICY CONTEXT and RESPONSE "
        "provided.\n\n"

        "Respond with ONLY a JSON object:\n"
        '{"decision": "RESOLVE" or "ESCALATE", '
        '"reason": "<short explanation citing the policy>"}'
    )

    user_prompt = (
        f"POLICY CONTEXT:\n{policy_context}\n\n"
        f"AGENT RESPONSE:\n"
        f"{state.get('generated_response', '')}\n\n"
        f"EMPLOYEE REQUEST:\n"
        f"{state['original_request']}"
    )

    fallback = {
        "decision": "ESCALATE",
        "reason": (
            "Could not determine a confident decision; "
            "escalating for human review."
        ),
    }

    result = call_llm_json(
        system_prompt,
        user_prompt,
        fallback,
    )

    decision = result.get("decision", "ESCALATE")

    if decision not in ("RESOLVE", "ESCALATE"):
        decision = "ESCALATE"

    return {
        "decision": decision,
        "decision_reason": result.get("reason", ""),
    }


def route_after_decision(state):
    """RESOLVE goes to audit; ESCALATE creates a ticket first."""

    return (
        "create_ticket"
        if state.get("decision") == "ESCALATE"
        else "audit_log"
    )


# ---------------------------------------------------------------------------
# 6. Terminal / escalation-path nodes
# ---------------------------------------------------------------------------

def no_policy_node(state):
    """No relevant policy was found - escalate without generation."""

    return {
        "generated_response": NO_POLICY_MESSAGE,
        "grounded": False,
        "grounding_reason": (
            "No relevant policy retrieved above the relevance threshold."
        ),
        "decision": "ESCALATE",
        "decision_reason": (
            "No Veridian Corp policy covers this request; "
            "requires human review."
        ),
    }


def grounding_failed_node(state):
    """Grounding failed - escalate without regeneration or retry."""

    return {
        "generated_response": GROUNDING_FAIL_MESSAGE,
        "decision": "ESCALATE",
        "decision_reason": (
            f"Grounding check failed: "
            f"{state.get('grounding_reason', '')}"
        ),
    }


def create_ticket_node(state):
    """Create a ticket in SQLite for an escalated request."""

    ticket_id = create_ticket(
        request_id=state.get("request_id", ""),
        employee_id=state.get("employee_id", ""),
        employee_name=state.get("employee_name", ""),
        employee_email=state.get("employee_email", ""),
        category=state.get("category", "Other"),
        issue=state.get("original_request", ""),
        reason=state.get("decision_reason", ""),
        status="ESCALATED",
    )

    return {
        "ticket_id": ticket_id
    }


# ---------------------------------------------------------------------------
# 7. Audit
# ---------------------------------------------------------------------------

def audit_log_node(state):
    """Write the final audit trail row."""

    policy_used = ", ".join(
        p["source"]
        for p in state.get("retrieved_policies", [])
    ) or "None"

    ticket_context = ", ".join(
        t["ticket_id"]
        for t in state.get("retrieved_tickets", [])
    ) or "None"

    entry = {
        "request_id": state.get("request_id"),
        "intent": state.get("intent"),
        "category": state.get("category"),
        "policy_used": policy_used,
        "ticket_context": ticket_context,
        "grounding_result": (
            "PASS"
            if state.get("grounded")
            else "FAIL"
        ),
        "grounding_reason": state.get(
            "grounding_reason",
            "",
        ),
        "decision": state.get("decision"),
        "action": state.get("decision"),
        "ticket_id": state.get("ticket_id"),
        "response": state.get("generated_response"),
    }

    log_audit(entry)

    return {
        "audit_data": entry
    }