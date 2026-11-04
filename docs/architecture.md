
# CloudDesk AI Support Copilot — Architecture

```mermaid
flowchart TD
    U["User / Support Reviewer"] --> UI["Streamlit Dashboard"]
    UI --> R["Smart LangGraph Router"]

    R --> D["Documentation Questions"]
    R --> T["Ticket Status Questions"]
    R --> A["Draft Requests"]
    R --> C["Clarification / Unsupported"]

    D --> E["EmbeddingGemma"]
    E --> V[("ChromaDB Knowledge Base")]
    V --> L["Gemma 3 Answer Generation"]
    L --> UI

    T --> J[("Fictional Tickets JSON")]
    J --> UI

    A --> W["LangGraph Approval Workflow"]
    W --> K["Ticket Lookup"]
    K --> Q["Retrieve Relevant Policies"]
    Q --> G["Gemma 3 Draft Generation"]
    G --> H{"Human Approval"}

    H -->|Reject| X["End — No Save"]
    H -->|Approve| S["Idempotent SQLite Save"]

    S --> P[("Approved Reply Records")]
    S --> F["JSONL Compatibility Export"]

    W -.-> CP[("SQLite Checkpoints")]
    CP -.-> REC["Recover Pending Approval"]
    REC --> H

    C --> UI
    X --> UI
    P --> UI
```

## Key Components

- **Streamlit:** User-facing application and approval interface.
- **Smart LangGraph Router:** Classifies requests and selects a workflow.
- **Ollama / Gemma 3:** Runs local language-model inference.
- **EmbeddingGemma:** Produces document and query embeddings.
- **ChromaDB:** Stores and retrieves company-policy documents.
- **Ticket Lookup:** Reads fictional customer support tickets.
- **Human Approval:** Interrupts drafting until a reviewer approves or rejects.
- **SQLite Checkpoints:** Persist workflow state across application restarts.
- **SQLite Approval Records:** Prevent duplicate saves using unique approval IDs.
- **JSONL Export:** Provides a local export of approved replies.

## Safety Boundaries

- No automatic email sending.
- Human approval is required before a draft is saved as approved.
- Approved replies have the status `approved_not_sent`.
- Local databases and approval records are excluded from Git.
- This is a portfolio demonstration, not a production customer-support deployment.
