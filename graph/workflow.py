"""
Builds the LangGraph StateGraph for the Veridian IT Service Agent.

    START
      |
      v
  analyze_request
      |
      v
  retrieve_context
      |
      +-- policy_found? --NO--> no_policy
      |                              |
     YES                             v
      |                         create_ticket
      v                              |
  generate_response                  |
      |                              |
      v                              |
  check_grounding                    |
      |                              |
      +-- grounded? --NO--> grounding_failed
      |                              |
     YES                             |
      |                              |
      v                              |
  decide_action                     |
      |                              |
      +-- ESCALATE ------------------+
      |
    RESOLVE
      |
      v
   audit_log --> END

No follow-up questions.
No retry loop.
No query rewriting.
No regeneration by design.
"""

from langgraph.graph import StateGraph, END

from graph.state import AgentState

from graph.nodes import (
    analyze_request,
    retrieve_context,
    route_after_retrieve,
    generate_response,
    check_grounding,
    route_after_grounding,
    decide_action,
    route_after_decision,
    no_policy_node,
    grounding_failed_node,
    create_ticket_node,
    audit_log_node,
)


def build_workflow():

    graph = StateGraph(AgentState)

    # -----------------------------------------------------------------------
    # Nodes
    # -----------------------------------------------------------------------

    graph.add_node("analyze_request", analyze_request)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_response", generate_response)
    graph.add_node("check_grounding", check_grounding)
    graph.add_node("decide_action", decide_action)
    graph.add_node("no_policy", no_policy_node)
    graph.add_node("grounding_failed", grounding_failed_node)
    graph.add_node("create_ticket", create_ticket_node)
    graph.add_node("audit_log", audit_log_node)

    # -----------------------------------------------------------------------
    # Entry
    # -----------------------------------------------------------------------

    graph.set_entry_point("analyze_request")

    # -----------------------------------------------------------------------
    # Analysis -> Retrieval
    # -----------------------------------------------------------------------

    graph.add_edge(
        "analyze_request",
        "retrieve_context"
    )

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve_context",
        route_after_retrieve,
        {
            "generate_response": "generate_response",
            "no_policy": "no_policy",
        },
    )

    # -----------------------------------------------------------------------
    # Generation -> Grounding
    # -----------------------------------------------------------------------

    graph.add_edge(
        "generate_response",
        "check_grounding"
    )

    # -----------------------------------------------------------------------
    # Grounding
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(
        "check_grounding",
        route_after_grounding,
        {
            "decide_action": "decide_action",
            "grounding_failed": "grounding_failed",
        },
    )

    # -----------------------------------------------------------------------
    # Decision
    # -----------------------------------------------------------------------

    graph.add_conditional_edges(
        "decide_action",
        route_after_decision,
        {
            "create_ticket": "create_ticket",
            "audit_log": "audit_log",
        },
    )

    # -----------------------------------------------------------------------
    # Escalation paths
    # -----------------------------------------------------------------------

    graph.add_edge(
        "no_policy",
        "create_ticket"
    )

    graph.add_edge(
        "grounding_failed",
        "create_ticket"
    )

    graph.add_edge(
        "create_ticket",
        "audit_log"
    )

    # -----------------------------------------------------------------------
    # Final node
    # -----------------------------------------------------------------------

    graph.add_edge(
        "audit_log",
        END
    )

    return graph.compile()


_compiled_workflow = None


def get_workflow():
    global _compiled_workflow

    if _compiled_workflow is None:
        _compiled_workflow = build_workflow()

    return _compiled_workflow


def run_workflow(
    request_id: str,
    employee_name: str,
    employee_id: str,
    employee_email: str,
    original_request: str,
) -> dict:
    """Run the full graph for a single employee request."""

    workflow = get_workflow()

    initial_state = {
        "request_id": request_id,
        "employee_name": employee_name,
        "employee_id": employee_id,
        "employee_email": employee_email,
        "original_request": original_request,
        "ticket_id": None,
    }

    final_state = workflow.invoke(initial_state)

    return final_state