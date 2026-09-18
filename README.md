# Veridian Corp — Internal IT Service Agent

A prototype agentic AI system that receives employee IT requests, retrieves relevant Veridian Corp policy and ticket context, generates a grounded response, checks that response for hallucinations, decides whether to resolve or escalate, creates a ticket when needed, and records an audit trail.

Built for the **AIONOS Agentic AI Factory — Assignment 2: Internal Service Agent (IT Support)**.

---

## 1. Project Overview

Employees at Veridian Corp submit IT requests in plain English (password lockouts, VPN access, laptop issues, software installs, security incidents, etc.). This agent reads each request, identifies the intent and category, retrieves the applicable company policy and existing ticket history for context, generates a response using the retrieved information, checks the response for grounding, and decides whether the request can be resolved or should be escalated.

The system is intentionally designed to use **only the supplied Veridian Corp Data Pack as its source data**. When the available policy information is insufficient, the agent escalates instead of inventing a company rule.

---

## 2. Problem Statement

Internal IT support receives repetitive, policy-driven requests that could be handled consistently if a system could:

1. Understand what the employee is asking.
2. Retrieve the relevant company policy.
3. Use existing ticket information as supporting context.
4. Generate responses grounded in the retrieved policy.
5. Detect when human intervention is required.
6. Create an IT ticket when escalation is necessary.
7. Maintain an audit trail of the request and agent decision.

---

## 3. Architecture

```text
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
    v           v
No Policy    Generate
    |         Response
    |           |
    |           v
    |     Grounding Check
    |        /        \
    |      FAIL       PASS
    |       |           |
    |       v           v
    |   Escalate    Decide Action
    |                 /        \
    |             RESOLVE    ESCALATE
    |                |           |
    |                |           v
    |                |       Create Ticket
    |                |           |
    +----------------+-----------+
                     |
                     v
                 Audit Log
                     |
                    END
```

The workflow is implemented using **LangGraph**.

**There is no retry loop, query rewriting, follow-up clarification, or regeneration.** If grounding fails, the agent escalates immediately rather than trying to generate another answer. If no relevant policy is found, it also escalates instead of guessing.

---

## 4. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| LLM | Groq API — `openai/gpt-oss-20b` |
| Orchestration | LangGraph |
| RAG framework | LangChain |
| Embeddings | FastEmbed (local) |
| Vector store | ChromaDB (local, persisted to disk) |
| Agent state | TypedDict |
| Relational storage | SQLite |
| Data processing | Pandas |
| Frontend | Streamlit |
| Validation | Pydantic |
| Testing | Pytest |
| Version Control | Git + GitHub |

The prototype intentionally avoids external ticketing APIs, Redis, Postgres/MongoDB, multi-agent frameworks, authentication, and other infrastructure not required for the assignment.

---

## 5. Project Structure

```text
veridian-it-agent/
│
├── app.py                         # Streamlit UI
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── graph/
│   ├── state.py                   # LangGraph AgentState
│   ├── nodes.py                   # Agent workflow node functions
│   └── workflow.py                # StateGraph wiring + conditional routing
│
├── rag/
│   ├── ingest.py                  # Builds ChromaDB policy index
│   └── retriever.py               # Policy semantic search + ticket search
│
├── llm/
│   └── client.py                  # Groq client wrapper
│
├── database/
│   └── db.py                      # SQLite schema, requests, tickets, audit logs
│
├── data/
│   ├── policies/                  # Veridian Corp policy documents
│   ├── requests.csv               # Assignment employee requests
│   └── tickets.csv                # Assignment existing ticket queue
│
└── tests/
    └── test_core.py               # Core workflow tests
```

---

## 6. Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Vansh3829/AIonos_Veridian_IT_Assistent.git
cd AIonos_Veridian_IT_Assistent
```

### 2. Create a virtual environment

```bash
python3.12 -m venv venv
```

Activate it:

**macOS/Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then set:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The `.env` file is intentionally excluded from Git.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

The SQLite database and ChromaDB index are initialized automatically when required.

---

## 7. Environment Variables

The application uses the following configuration:

```dotenv
GROQ_API_KEY=

GROQ_MODEL=openai/gpt-oss-20b
LLM_TEMPERATURE=0.0

# --- Embeddings run locally ---
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# --- Vector store ---
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=veridian_policies

# --- Policy ingestion / retrieval ---
CORPUS_DIR=./data/policies
CHUNK_SIZE=800
CHUNK_OVERLAP=120
TOP_K=4
RELEVANCE_THRESHOLD=0.55
```

| Variable | Purpose |
|---|---|
| `GROQ_MODEL` | Groq-hosted LLM used by the agent |
| `LLM_TEMPERATURE` | LLM sampling temperature |
| `EMBEDDING_MODEL` | Local FastEmbed model |
| `CHROMA_PERSIST_DIR` | ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name |
| `CORPUS_DIR` | Directory containing policy documents |
| `CHUNK_SIZE` | Maximum policy chunk size |
| `CHUNK_OVERLAP` | Chunk overlap |
| `TOP_K` | Number of policy results retrieved |
| `RELEVANCE_THRESHOLD` | Minimum relevance score required for a policy to count as relevant |

---

## 8. How Policy Ingestion Works

The application builds the ChromaDB index automatically when an index is not already available.

The policy pipeline is:

```text
Policy Markdown Files
        |
        v
Document Loading
        |
        v
Text Chunking
        |
        v
FastEmbed Embeddings
        |
        v
ChromaDB
```

The policy files are stored under:

```text
data/policies/
```

To manually rebuild the index:

```bash
python -m rag.ingest
```

FastEmbed downloads the embedding model the first time it is required. After the model is cached, it can be reused locally.

---

## 9. How the RAG Pipeline Works

### Policy Retrieval

`rag/retriever.py` performs semantic similarity search against the ChromaDB collection.

The retriever:

1. Receives the employee request.
2. Searches the policy embeddings.
3. Returns the top `TOP_K` results.
4. Calculates relevance scores.
5. Applies `RELEVANCE_THRESHOLD`.
6. Marks the request as having a relevant policy only when at least one result meets the threshold.

With the current configuration:

```text
TOP_K = 4
RELEVANCE_THRESHOLD = 0.55
```

If no retrieved policy reaches the threshold, the agent follows the **No Policy** branch and escalates.

### Ticket Retrieval

Existing tickets are stored in the `ticket_queue` table.

Because the provided queue contains only a small number of short records, ticket retrieval uses keyword/category matching rather than vector search.

Active tickets are prioritized over closed historical tickets.

---

## 10. How Grounding Check Works

After the LLM generates a response, the agent performs a separate grounding check.

The grounding checker receives:

- Retrieved policy context
- Generated response

It evaluates whether the generated response is supported by the retrieved policy information.

The result contains:

```text
grounded
reason
```

### Grounding PASS

```text
Generate Response
       |
       v
Grounding PASS
       |
       v
Decide Action
```

### Grounding FAIL

```text
Generate Response
       |
       v
Grounding FAIL
       |
       v
Escalate
       |
       v
Create Ticket
       |
       v
Audit Log
```

There is **no regeneration or retry** after a grounding failure.

---

## 11. How Escalation Works

The agent escalates when human intervention is required.

### Case 1 — No Relevant Policy

If the retrieved policies do not meet the relevance threshold, the agent does not invent a procedure.

It returns a safe no-policy response and creates an escalation ticket.

### Case 2 — Grounding Failure

If the generated response contains information that is not sufficiently supported by the retrieved policy context, the agent escalates immediately.

### Case 3 — Policy Requires Human Action

Some policies require actions or approvals involving another person/team, such as:

- Manager approval
- Finance processing
- IT Security review
- Security incident handling

In these cases the agent routes the request appropriately rather than claiming it can complete the process itself.

---

## 12. Ticket Creation

Agent-created tickets are stored separately from the original Data Pack ticket queue.

Tickets use sequential IDs:

```text
TKT-1001
TKT-1002
TKT-1003
...
```

The ticket contains:

```text
ticket_id
request_id
employee_id
employee_name
employee_email
category
issue
reason
status
created_at
```

Escalated tickets are stored with:

```text
status = ESCALATED
```

The original Data Pack ticket queue remains available as reference context.

---

## 13. Audit Logging

Each processed request generates an audit record.

The audit trail captures information such as:

- Request ID
- Timestamp
- Intent
- Category
- Relevant policy
- Ticket context
- Grounding result
- Grounding reason
- Decision
- Created ticket ID, when applicable
- Final response

The audit information can be viewed through the Streamlit interface.

---

## 14. Example Demo Scenarios

### Scenario 1 — Password Lockout

**Request**

```text
I'm locked out of my account, I tried my password 6 times.
```

**Relevant policy:** KB-01 Password Reset

The policy states that after 5 failed attempts the account is locked and the employee should contact IT for a manual unlock. No approval is required.

**Expected action:** Resolve

---

### Scenario 2 — Contractor VPN

**Request**

```text
New contractor joining my team next week, they'll need VPN access.
```

**Relevant policy:** KB-02 VPN Access

The policy states that contractors require manager approval through the access request form.

**Expected action:** Escalate and create a ticket.

---

### Scenario 3 — Phishing Email

**Request**

```text
I think I got a phishing email asking for my login.
```

**Relevant policy:** KB-09 Security Incident

The policy requires suspected phishing incidents to be reported immediately to:

```text
security@veridian-corp.example
```

The email should not be forwarded to other employees.

**Expected action:** Escalate and create a ticket.

---

### Scenario 4 — Unsupported Request

**Request**

```text
My internet is not working.
```

The supplied policy pack does not establish a specific internet-connectivity troubleshooting procedure.

The agent should not invent one.

**Expected action:** Escalate and create a ticket.

---

### Scenario 5 — General Request

**Request**

```text
hey can you help, its not working
```

The agent does not use a follow-up/clarification loop. If the available policy context cannot safely support an answer, the request follows the escalation path rather than the agent inventing the missing details.

**Expected action:** Escalate.

---

## 15. Data Sources and Grounding

This prototype uses the supplied **Veridian Corp Assignment 2 Data Pack** as its source data.

The source data includes:

- Company policies
- Employee requests
- Existing ticket queue

The agent does **not** use external web search or external enterprise ticketing systems to answer requests.

This is intentional: if the provided source data does not contain a sufficiently relevant policy, the agent escalates instead of generating an unsupported company rule.

---

## 16. Assignment Assumptions

- Policy, employee request, and ticket data are based on the supplied Assignment 2 Data Pack.
- The agent does not infer a company-wide policy from a single historical ticket outcome.
- Historical tickets are used as context and precedent, not automatically treated as policy.
- When multiple applicable policies are retrieved, the response should consider the relevant policy context rather than arbitrarily ignoring conflicting or additional requirements.
- The system is a prototype and does not represent a production enterprise IT service desk.
- Agent-created tickets are stored locally in SQLite.

---

## 17. Security

The Groq API key is not committed to GitHub.

The project uses:

```text
.env
```

for local secrets, and `.env` is excluded through `.gitignore`.

For Streamlit Community Cloud, the Groq API key should be stored using **Streamlit Secrets**.

Never place a real API key inside:

- `README.md`
- `.env.example`
- GitHub source files
- screenshots
- presentation slides

---

## 18. Streamlit Community Cloud Deployment

The application can be deployed directly from GitHub using Streamlit Community Cloud.

### Configuration

```text
Repository:
Vansh3829/AIonos_Veridian_IT_Assistent

Branch:
main

Main file:
app.py

Python:
3.12
```

### Streamlit Secrets

Add the following configuration in the Streamlit deployment settings:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-20b"
LLM_TEMPERATURE = "0.0"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "veridian_policies"

CORPUS_DIR = "./data/policies"
CHUNK_SIZE = "800"
CHUNK_OVERLAP = "120"
TOP_K = "4"
RELEVANCE_THRESHOLD = "0.55"
```

The actual API key should never be committed to the repository.

---

## 19. Testing

Run the test suite with:

```bash
pytest tests/ -v
```

The tests cover core workflow behavior such as:

- Policy retrieval
- No-policy handling
- Grounding PASS/FAIL
- No regeneration after grounding failure
- Resolve/ESCALATE decisions
- Ticket creation
- Sequential ticket IDs
- Audit logging

---

## 20. Limitations

This is a prototype rather than a production IT service management system.

Current limitations include:

- No authentication
- No real enterprise ticketing integration
- No external ITSM system
- No multi-turn conversation memory
- Local SQLite storage
- Local ChromaDB persistence
- Single-agent workflow
- LLM-dependent intent analysis and decision making
- No web search or external knowledge retrieval

The system intentionally prioritizes grounded responses and escalation over unsupported automation.

---

## 21. Main Files to Understand Before the Demo

### `graph/nodes.py`

Contains the main agent logic:

- Request analysis
- Context retrieval
- Response generation
- Grounding check
- Action decision
- Ticket creation
- Audit logging

### `graph/workflow.py`

Defines the LangGraph workflow and conditional routing between agent states.

### `rag/retriever.py`

Contains the policy retrieval logic and relevance threshold used to determine whether a policy is sufficiently relevant.

### `database/db.py`

Contains the SQLite tables and CRUD operations for requests, tickets, ticket queue data, and audit logs.

### `app.py`

Contains the Streamlit UI and invokes the LangGraph workflow.

---

## 22. Assignment Requirements Covered

- ✅ Working agent prototype
- ✅ Employee request analysis
- ✅ Policy retrieval
- ✅ Existing ticket context
- ✅ Grounded response generation
- ✅ Grounding validation
- ✅ Resolve / Escalate decision
- ✅ Ticket creation
- ✅ Audit logging
- ✅ Architecture/process flow
- ✅ Inputs and sources
- ✅ Assumptions
- ✅ AI tools used
- ✅ GitHub repository
- ✅ Streamlit deployment

---

## GitHub

Repository:

https://github.com/Vansh3829/AIonos_Veridian_IT_Assistent
