"""
Shared state passed between LangGraph nodes.

Kept flat and minimal - only fields the nodes actually read or write.

Employee information is entered by the employee:
- employee_name
- employee_id
- employee_email
- original_request

The request_id is generated automatically by the system.

No follow-up, retry_count, query-rewriting fields, or regeneration fields:
this workflow has no follow-up/retry/regeneration loops by design.
"""

from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict, total=False):

    # -----------------------------------------------------------------------
    # Employee Request Input
    # -----------------------------------------------------------------------

    request_id: str
    employee_name: str
    employee_id: str
    employee_email: str
    original_request: str


    # -----------------------------------------------------------------------
    # analyze_request output
    # -----------------------------------------------------------------------

    intent: str
    category: str
    extracted_details: str
    keywords: List[str]


    # -----------------------------------------------------------------------
    # retrieve_context output
    # -----------------------------------------------------------------------

    retrieved_policies: List[Dict[str, Any]]
    retrieved_tickets: List[Dict[str, Any]]
    policy_found: bool


    # -----------------------------------------------------------------------
    # generate_response output
    # -----------------------------------------------------------------------

    generated_response: str


    # -----------------------------------------------------------------------
    # check_grounding output
    # -----------------------------------------------------------------------

    grounded: bool
    grounding_reason: str


    # -----------------------------------------------------------------------
    # decide_action output
    # -----------------------------------------------------------------------

    decision: str
    decision_reason: str


    # -----------------------------------------------------------------------
    # create_ticket output
    # -----------------------------------------------------------------------

    ticket_id: Optional[str]


    # -----------------------------------------------------------------------
    # Final audit record
    # -----------------------------------------------------------------------

    audit_data: Dict[str, Any]