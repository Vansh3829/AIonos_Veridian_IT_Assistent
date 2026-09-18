# Veridian Corp — Internal IT Service Agent

A prototype agentic AI system that receives employee IT requests, retrieves relevant
Veridian Corp policy and ticket context, generates a grounded response, checks that
response for hallucinations, decides whether to resolve or escalate, creates a ticket
when needed, and records a full audit trail.

Built for the AIONOS Agentic AI Factory assignment.

---

## 1. Project Overview

Employees at Veridian Corp submit IT requests in plain English (password lockouts,
VPN access, laptop issues, software installs, etc.). This agent reads each request,
figures out what it's about, looks up the **actual company policy** that applies (if
any), and the **existing ticket history** for context, then answers the employee —
but only using what the retrieved documents actually say. Every answer is checked
for grounding before it's shown, and every decision is logged.

## 2. Problem Statement

Internal IT support gets repetitive, policy-driven requests that could be answered
instantly and consistently if a system could:
1. Understand what's being asked,
2. Look up the one true source of company policy,
3. Answer only from that source (no guessing/hallucinating rules that don't exist),
4. Know when a human has to be involved (approvals, security review, ambiguous
   requests), and
5. Keep a clear audit trail of what it decided and why.

## 3. Architecture

```
Employee Request
      |
      v
Analyze Request
      |
      v
Retrieve Context
(Policy + Existing Tickets)
      |
      v
Relevant Policy Found?
      /       \
    NO         YES
    |           |
No Relevant   Generate
Policy        Response
    |             |
    |     Grounding / Hallucination Check
    |          /          \
    |       FAIL          PASS
    |        |             |
    |    Escalate     Decide Action
    |        |          /          \
    |        |     RESOLVE       ESCALATE
    |        |         |             |
    |        |      Response    Create Ticket
    |        |         |             |
    +--> Create Ticket |             |
              |         |             |
              +----> Audit Log <-----+
                          |
                         END
```

**There is no retry loop, no query rewriting, and no regeneration.** If grounding
fails, the agent escalates immediately rather than trying again — this is a
deliberate safety-first design choice (see Section 12).

## 4. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Groq API — Llama 3.1 8B Instant |
| Orchestration | LangGraph |
| RAG framework | LangChain |
| Embeddings | FastEmbed (local, no API calls) |
| Vector store | ChromaDB (local, persisted to disk) |
| Structured data | Pydantic-style TypedDict state |
| Relational storage | SQLite |
| Data processing | Pandas |
| Frontend | Streamlit |

No Redis, no Postgres/Mongo, no external ticketing APIs, no multi-agent framework,
no auth system, no Docker — kept intentionally lightweight per the assignment brief.

## 5. Project Structure

```
veridian-it-agent/
├── app.py                     # Streamlit UI
├── graph/
│   ├── state.py                # LangGraph AgentState (TypedDict)
│   ├── nodes.py                 # All node functions (analyze, retrieve, generate, ground, decide, ticket, audit)
│   └── workflow.py              # StateGraph wiring + conditional edges
├── rag/
│   ├── ingest.py                 # Builds the ChromaDB index from data/policies/*.md
│   └── retriever.py              # Policy semantic search + ticket keyword search
├── llm/
│   └── client.py                  # Groq client wrapper + JSON-safe LLM call helper
├── database/
│   └── db.py                      # SQLite schema, ticket queue loader, ticket/audit CRUD
├── data/
│   ├── policies/                   # KB-01..KB-10 + Asset Management Policy (verbatim from Data Pack)
│   ├── requests.csv                 # REQ-01..REQ-15 (Data Pack Section 2)
│   └── tickets.csv                  # TK-1042..TK-1051 existing ticket queue (Data Pack Section 3)
├── tests/
│   └── test_core.py                  # Core logic tests (grounding, decisions, tickets, audit)
├── requirements.txt
├── .env.example
└── README.md
```

## 6. Setup Instructions

```bash
# 1. Clone / unzip the project, then cd into it
cd veridian-it-agent

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your Groq API key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

The app initializes the SQLite database and builds the ChromaDB index
automatically on first run — no separate setup script is required.

## 7. Environment Variables

`.env` (not committed — see `.env.example`):

```dotenv
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
LLM_TEMPERATURE=0.0

# --- Embeddings run 100% locally, no key needed ---
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# --- Vector store ---
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=veridian_policies

# --- Ingestion / retrieval ---
CORPUS_DIR=./data/policies
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
RELEVANCE_THRESHOLD=0.35
```

Get a free Groq key at https://console.groq.com. Everything else has a sensible
default baked in (matching the values above) — you only *have* to set
`GROQ_API_KEY`; the rest are there so the model, chunking, retrieval depth, and
grounding threshold can all be tuned without touching code:

| Variable | Used in | Purpose |
|---|---|---|
| `GROQ_MODEL` | `llm/client.py` | Which Groq-hosted model to call |
| `LLM_TEMPERATURE` | `llm/client.py` | Default sampling temperature (0.0 = deterministic) |
| `EMBEDDING_MODEL` | `rag/ingest.py`, `rag/retriever.py` | FastEmbed model for local embeddings |
| `CHROMA_PERSIST_DIR` | `rag/ingest.py`, `rag/retriever.py` | Where the vector index is stored on disk |
| `CHROMA_COLLECTION_NAME` | `rag/ingest.py`, `rag/retriever.py` | Chroma collection name |
| `CORPUS_DIR` | `rag/ingest.py` | Folder of policy `.md` files to ingest |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `rag/ingest.py` | Text-splitter settings (only kicks in if a policy file exceeds `CHUNK_SIZE`; every KB file here is short enough to stay a single chunk) |
| `TOP_K` | `rag/retriever.py` | How many policy chunks to retrieve per query |
| `RELEVANCE_THRESHOLD` | `rag/retriever.py` | Minimum Chroma relevance score (0–1) for a policy to count as "found" |

## 8. How to Ingest Policies

The Streamlit app auto-builds the ChromaDB index the first time it runs
(`rag/retriever.py` calls `rag/ingest.py:build_index()` if no index exists on
disk at `chroma_db/`). To rebuild it manually (e.g. after editing a policy file):

```bash
python -m rag.ingest
```

> **Note:** FastEmbed downloads its embedding model (`BAAI/bge-small-en-v1.5`)
> from Hugging Face the first time it runs. This requires internet access once;
> the model is then cached locally and subsequent runs work offline.

## 9. How to Run Streamlit

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar/dropdown to load any of
REQ-01 through REQ-15, or type a custom request into the text area.

## 10. How the RAG Pipeline Works

1. **Ingestion** (`rag/ingest.py`): Each policy file in `data/policies/` (one
   Markdown file per KB article — they're already short and self-contained) is
   loaded as a single document and embedded with FastEmbed, then stored in a
   persisted ChromaDB collection (`chroma_db/`).
2. **Retrieval** (`rag/retriever.py`):
   - **Policies:** `retrieve_policies(query)` runs a semantic similarity search
     against ChromaDB and returns the top `TOP_K` chunks (default 4) with
     relevance scores. `RELEVANCE_THRESHOLD` (default 0.35, both set in `.env`)
     — only chunks scoring at or above this are treated as "relevant enough" to
     answer from. Below that, `policy_found` is `False` and the agent refuses
     to guess.
   - **Tickets:** `retrieve_tickets(keywords)` is a simple keyword match over the
     `ticket_queue` SQLite table (loaded from `data/tickets.csv`), no vector
     search needed for 10 short rows. Active tickets are ranked above closed ones.

## 11. How Grounding Check Works

After `generate_response` produces an answer, `check_grounding` sends the
**retrieved policy context** and the **generated response** back to the LLM with
a strict instruction: flag anything in the response not directly supported by
the context. The result is a JSON object `{"grounded": bool, "reason": str}`.

- `grounded: true` → proceed to `decide_action`.
- `grounded: false` → **no retry, no regeneration.** The agent immediately
  escalates, creates a ticket, and logs the failure with the checker's reason.

## 12. How Escalation Works

Escalation is triggered in exactly three places:
1. **No relevant policy found** — the request doesn't match any KB article
   above the relevance threshold. The agent returns a fixed safe message and
   escalates rather than guessing.
2. **Grounding check fails** — the generated response wasn't fully supported
   by the retrieved policy. Escalate immediately, no regeneration attempt.
3. **`decide_action` returns ESCALATE`** — the policy itself says the request
   needs someone other than IT (manager approval, Finance, Security review) or
   isn't something IT can resolve unilaterally.

This "escalate rather than guess" design is the core safety mechanism of the
whole system — it trades a small amount of automation for a strong guarantee
against inventing company policy.

## 13. How Ticket Creation Works

`database/db.py:create_ticket()` inserts a row into the `tickets` table (separate
from the read-only `ticket_queue` reference table loaded from the Data Pack) with
a sequential ID (`TKT-1001`, `TKT-1002`, ...). Fields: `ticket_id`, `request_id`,
`employee`, `category`, `issue`, `reason`, `status`, `created_at`.

## 14. How Audit Logging Works

`database/db.py:log_audit()` writes one row per processed request to `audit_logs`,
capturing: request ID, timestamp, intent, category, which policy was used, which
tickets were referenced as context, the grounding result (PASS/FAIL) and reason,
the decision (RESOLVE/ESCALATE), the ticket ID if one was created, and the final
response text. Visible in the Streamlit sidebar and as JSON after each run.

## 15. Example Demo Scenarios

| # | Scenario | Sample Request | Expected Outcome |
|---|---|---|---|
| 1 | Password Lockout | *"I'm locked out of my account, tried my password 6 times."* (REQ-03) | Retrieves KB-01, grounded response, **RESOLVE** (IT manually unlocks, no approval needed) |
| 2 | Contractor VPN | *"New contractor joining my team next week, they'll need VPN access."* (REQ-11) | Retrieves KB-02, grounded, **ESCALATE** (manager approval required for contractors) → ticket created |
| 3 | Phishing | *"I think I got a phishing email asking for my login..."* (REQ-08) | Retrieves KB-09, grounded, **ESCALATE** (must report to Security immediately) → ticket created |
| 4 | Unsupported request | *"Requesting approval to install a browser extension for productivity tracking."* (REQ-14) | No policy directly authorizes/covers this beyond generic software review — agent does not invent a browser-extension policy; if no policy clears the relevance threshold, returns "no relevant policy" and escalates for human review |
| 5 | Ambiguous request | *"hey can you help, its not working"* (REQ-15) | `sufficient_info = false` from `analyze_request` → agent does not guess, asks for clarification / escalates for human review |

## 16. Limitations

- This is a prototype, not a production IT system — no authentication, no real
  ticketing integration, no multi-turn conversation memory.
- Grounding and decision quality depend on the underlying LLM (Llama 3.1 8B via
  Groq); it is fast and cheap but less capable than larger models. The strict
  system prompts and the grounding-check safety net exist specifically to
  compensate for this.
- The relevance threshold (0.35) was chosen empirically for this small, 11-document
  knowledge base and may need adjustment for a larger real-world KB.
- Ticket/audit data is stored in a local SQLite file (`veridian.db`), which is
  fine for a demo but not for concurrent multi-user production use.

## 17. Assignment Assumptions

- All policy, request, and ticket data is taken **verbatim** from the supplied
  Assignment 2 Data Pack. Nothing was invented or extrapolated beyond it.
- Where the Data Pack does not establish a rule (e.g. no explicit browser-extension
  policy, no explicit admin-access-to-finance-server policy), the agent is
  designed to say so and escalate rather than infer a rule from a single past
  ticket outcome (e.g. TK-1050 being rejected does not, by itself, establish a
  blanket policy).
- For REQ-01 (3.5-year-old dead laptop), both KB-03 (3-year replacement /
  verified hardware failure) and the Asset Management Policy (4-year refresh
  cycle, Finance sign-off for early replacement) are retrieved as relevant
  context; the agent is instructed not to pick one and ignore the other, and
  the grounding checker will flag a response that oversimplifies this tension.

---

## Testing

```bash
pytest tests/ -v
```

Covers: relevant/no-relevant policy retrieval, grounding PASS/FAIL, **grounding
FAIL does not trigger regeneration** (the most important test), RESOLVE/ESCALATE
decisions, ticket creation (including sequential IDs), and audit logging.

## Main Files to Understand Before the Demo

1. **`graph/nodes.py`** — all the actual agent logic lives here, one function per
   workflow step. Start here.
2. **`graph/workflow.py`** — how the nodes are wired into a graph with conditional
   routing. Shows the escalate-vs-continue branches explicitly.
3. **`rag/retriever.py`** — the relevance threshold and retrieval logic that
   prevents the agent from answering without real policy backing.
4. **`database/db.py`** — the three tables and how tickets/audit rows are created.
5. **`app.py`** — thin UI layer that just calls `graph.workflow.run_workflow()`
   and displays the resulting state.
