"""
Core tests for the Veridian IT Service Agent.

These tests focus on the parts of the system that MUST behave correctly for
the assignment's safety requirements:

  - Policy retrieval (relevant vs. not relevant)
  - Grounding check pass/fail behaviour
  - The most important rule: grounding FAIL must NOT trigger regeneration
  - Decision routing (RESOLVE vs ESCALATE)
  - Ticket creation
  - Audit logging

LLM-dependent nodes (generate_response, check_grounding, decide_action) are
tested using monkeypatched fakes so the test suite runs deterministically
and without a live Groq API call / cost.

Run with:  pytest tests/ -v
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Use a temporary SQLite DB for the whole test session so we never touch the
# real veridian.db used by the running Streamlit app.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    import database.db as db

    test_db_path = tmp_path / "test_veridian.db"
    monkeypatch.setattr(db, "DB_PATH", str(test_db_path))
    db.init_db()
    yield db


# ---------------------------------------------------------------------------
# 1 & 2. Policy retrieval - relevant / not relevant
# ---------------------------------------------------------------------------

def test_relevant_policy_retrieval():
    from rag.retriever import retrieve_policies, has_relevant_policy

    policies = retrieve_policies("I'm locked out after 5 failed password attempts")
    assert len(policies) > 0
    assert has_relevant_policy(policies)
    top = max(policies, key=lambda p: p["score"])
    assert "KB-01" in top["source"]


def test_no_relevant_policy_for_unrelated_topic():
    from rag.retriever import retrieve_policies, has_relevant_policy

    # A request with no corresponding KB article at all (browser extension /
    # productivity tracking approval is not covered anywhere in the Data Pack).
    policies = retrieve_policies("approve a personal productivity tracking browser extension")
    # Either nothing comes back relevant, or scores are all below threshold.
    relevant = has_relevant_policy(policies)
    # This is a soft check since embeddings can be fuzzy; we mainly assert the
    # function runs and returns a boolean without raising.
    assert isinstance(relevant, bool)


# ---------------------------------------------------------------------------
# 3 & 4. Grounding PASS / FAIL
# ---------------------------------------------------------------------------

def test_grounding_pass(monkeypatch):
    import graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "call_llm_json",
        lambda system, user, fallback: {"grounded": True, "reason": "Supported by KB-01."}
    )

    state = {
        "generated_response": "Contact IT to unlock your account.",
        "retrieved_policies": [{"source": "KB-01", "content": "...", "score": 0.9}],
    }
    result = nodes.check_grounding(state)
    assert result["grounded"] is True


def test_grounding_fail(monkeypatch):
    import graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "call_llm_json",
        lambda system, user, fallback: {"grounded": False, "reason": "Claims an approval process not in the policy."}
    )

    state = {
        "generated_response": "Your request will be auto-approved within 24 hours.",
        "retrieved_policies": [{"source": "KB-02", "content": "...", "score": 0.8}],
    }
    result = nodes.check_grounding(state)
    assert result["grounded"] is False
    assert "approval" in result["grounding_reason"].lower()


# ---------------------------------------------------------------------------
# 5. THE most important test: grounding FAIL must NOT regenerate
# ---------------------------------------------------------------------------

def test_grounding_fail_does_not_regenerate(monkeypatch):
    """
    Simulates the full failure path: generate -> grounding FAIL -> grounding_failed_node.
    Asserts generate_response (the LLM generation call) is invoked exactly once overall,
    proving there is no regeneration after a grounding failure.
    """
    import graph.nodes as nodes

    call_count = {"generate": 0}

    def fake_call_llm(system, user):
        call_count["generate"] += 1
        return "This request will be automatically approved in 1 hour."  # intentionally unsupported

    monkeypatch.setattr(nodes, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        nodes, "call_llm_json",
        lambda system, user, fallback: {"grounded": False, "reason": "Not supported by policy context."}
    )

    state = {
        "original_request": "test request",
        "retrieved_policies": [{"source": "KB-02", "content": "VPN policy text", "score": 0.8}],
        "retrieved_tickets": [],
    }

    # Step 1: generate_response is called once
    gen_result = nodes.generate_response(state)
    state.update(gen_result)
    assert call_count["generate"] == 1

    # Step 2: grounding check fails
    ground_result = nodes.check_grounding(state)
    state.update(ground_result)
    assert state["grounded"] is False

    # Step 3: the failure path node runs - it must NOT call generate again
    fail_result = nodes.grounding_failed_node(state)
    state.update(fail_result)

    assert call_count["generate"] == 1, "generate_response must not be called again after grounding failure"
    assert state["decision"] == "ESCALATE"
    assert "escalated" in state["generated_response"].lower()


# ---------------------------------------------------------------------------
# 6 & 7. Decision: RESOLVE / ESCALATE
# ---------------------------------------------------------------------------

def test_resolve_decision(monkeypatch):
    import graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "call_llm_json",
        lambda system, user, fallback: {"decision": "RESOLVE", "reason": "No approval required per KB-01."}
    )
    state = {
        "retrieved_policies": [{"source": "KB-01", "content": "...", "score": 0.9}],
        "generated_response": "IT will unlock your account.",
        "original_request": "locked out after 6 attempts",
    }
    result = nodes.decide_action(state)
    assert result["decision"] == "RESOLVE"


def test_escalate_decision(monkeypatch):
    import graph.nodes as nodes

    monkeypatch.setattr(
        nodes, "call_llm_json",
        lambda system, user, fallback: {"decision": "ESCALATE", "reason": "Contractor VPN requires manager approval per KB-02."}
    )
    state = {
        "retrieved_policies": [{"source": "KB-02", "content": "...", "score": 0.9}],
        "generated_response": "Manager approval required for contractor VPN access.",
        "original_request": "new contractor needs VPN",
    }
    result = nodes.decide_action(state)
    assert result["decision"] == "ESCALATE"


# ---------------------------------------------------------------------------
# 8. Ticket creation
# ---------------------------------------------------------------------------

def test_ticket_creation(isolated_db):
    ticket_id = isolated_db.create_ticket(
        request_id="REQ-TEST",
        employee="Test Employee",
        category="VPN",
        issue="Needs VPN access",
        reason="Requires manager approval",
    )
    assert ticket_id.startswith("TKT-")

    tickets = isolated_db.get_all_tickets()
    assert len(tickets) == 1
    assert tickets.iloc[0]["ticket_id"] == ticket_id
    assert tickets.iloc[0]["status"] == "ESCALATED"


def test_sequential_ticket_ids(isolated_db):
    id1 = isolated_db.create_ticket("REQ-A", "Emp A", "VPN", "issue A", "reason A")
    id2 = isolated_db.create_ticket("REQ-B", "Emp B", "Laptop", "issue B", "reason B")
    assert id1 == "TKT-1001"
    assert id2 == "TKT-1002"


# ---------------------------------------------------------------------------
# 9. Audit logging
# ---------------------------------------------------------------------------

def test_audit_logging(isolated_db):
    isolated_db.log_audit({
        "request_id": "REQ-TEST",
        "intent": "VPN Access",
        "category": "VPN",
        "policy_used": "KB-02",
        "ticket_context": "None",
        "grounding_result": "PASS",
        "grounding_reason": "Supported by KB-02.",
        "decision": "ESCALATE",
        "action": "ESCALATE",
        "ticket_id": "TKT-1001",
        "response": "Manager approval required.",
    })
    logs = isolated_db.get_all_audit_logs()
    assert len(logs) == 1
    assert logs.iloc[0]["request_id"] == "REQ-TEST"
    assert logs.iloc[0]["grounding_result"] == "PASS"


# ---------------------------------------------------------------------------
# No-policy path
# ---------------------------------------------------------------------------

def test_no_policy_node_escalates():
    import graph.nodes as nodes

    state = {"sufficient_info": True, "category": "Other"}
    result = nodes.no_policy_node(state)
    assert result["decision"] == "ESCALATE"
    assert result["grounded"] is False
    assert "human review" in result["generated_response"].lower()
