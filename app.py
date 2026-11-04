
import re
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from smart_agent import ask_smart_agent
from approval_agent import approval_agent


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="CloudDesk AI Support Copilot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 CloudDesk AI Support Copilot")

st.write(
    "Ask CloudDesk questions, check support tickets, "
    "or generate customer replies with human approval."
)


# ==========================================
# 2. SESSION STATE
# ==========================================

DEFAULTS = {
    "approval_thread_id": None,
    "pending_draft": None,
    "pending_ticket_id": None,
    "approval_message": None,
    "last_answer": None,
    "last_intent": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def extract_ticket_id(text):
    match = re.search(
        r"\bTKT-\d+\b",
        text,
        re.IGNORECASE
    )
    return match.group(0).upper() if match else None


def get_workflow_config(thread_id):
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


def is_waiting_for_approval(snapshot):
    """
    Check whether LangGraph is paused at
    the human approval node.
    """
    return (
        snapshot is not None
        and "approval" in snapshot.next
    )


def clear_pending_draft():
    st.session_state.approval_thread_id = None
    st.session_state.pending_ticket_id = None
    st.session_state.pending_draft = None


# ==========================================
# 4. START DRAFT WORKFLOW
# ==========================================

def start_draft_workflow(ticket_id):

    if st.session_state.pending_draft is not None:
        st.warning(
            "Please approve or reject your current "
            "draft before starting another."
        )
        return

    ticket_id = ticket_id.strip().upper()

    if not re.fullmatch(r"TKT-\d+", ticket_id):
        st.warning(
            "Enter a valid ticket ID, such as TKT-1002."
        )
        return

    thread_id = str(uuid4())
    config = get_workflow_config(thread_id)

    with st.spinner("Generating customer reply..."):
        try:
            result = approval_agent.invoke(
                {"ticket_id": ticket_id},
                config=config
            )

            if result.get("__interrupt__"):

                snapshot = approval_agent.get_state(
                    config
                )

                if not is_waiting_for_approval(snapshot):
                    st.error(
                        "The workflow is not paused "
                        "at the approval step."
                    )
                    return

                draft = snapshot.values.get("draft", "")

                if not isinstance(draft, str) or not draft.strip():
                    st.error("No generated draft was found.")
                    return

                st.session_state.approval_thread_id = (
                    thread_id
                )
                st.session_state.pending_ticket_id = (
                    ticket_id
                )
                st.session_state.pending_draft = draft
                st.session_state.approval_message = None

                st.rerun()

            else:
                st.info(
                    result.get(
                        "result",
                        "Workflow completed."
                    )
                )

        except Exception as error:
            st.error("Draft generation failed.")
            st.code(
                f"{type(error).__name__}: {error}"
            )


# ==========================================
# 5. RECOVER SAVED WORKFLOW
# ==========================================

def recover_draft(thread_id):
    """
    Restore an interrupted approval workflow
    using its persistent SQLite checkpoint.
    """

    thread_id = thread_id.strip()

    if not thread_id:
        st.warning("Please enter a workflow ID.")
        return

    if st.session_state.pending_draft is not None:
        st.warning(
            "A draft is already open. Review it before "
            "recovering another workflow."
        )
        return

    config = get_workflow_config(thread_id)

    try:
        snapshot = approval_agent.get_state(config)

        if not is_waiting_for_approval(snapshot):
            st.warning(
                "No pending approval was found for "
                "this workflow ID. It may be invalid "
                "or already completed."
            )
            return

        values = snapshot.values

        draft = values.get("draft", "")
        ticket_id = values.get("ticket_id", "")

        if not isinstance(draft, str) or not draft.strip():
            st.error(
                "The checkpoint does not contain "
                "a valid draft."
            )
            return

        if not isinstance(ticket_id, str) or not ticket_id:
            st.error(
                "The checkpoint does not contain "
                "a valid ticket ID."
            )
            return

        st.session_state.approval_thread_id = thread_id
        st.session_state.pending_ticket_id = ticket_id
        st.session_state.pending_draft = draft
        st.session_state.approval_message = None

        st.rerun()

    except Exception as error:
        st.error("Checkpoint recovery failed.")
        st.code(
            f"{type(error).__name__}: {error}"
        )


# ==========================================
# 6. RESUME HUMAN APPROVAL
# ==========================================

def resume_approval(decision, edited_draft):

    thread_id = st.session_state.approval_thread_id

    if not thread_id:
        st.error("No approval workflow is selected.")
        return

    if decision == "approve" and not edited_draft.strip():
        st.error("Approved reply cannot be empty.")
        return

    config = get_workflow_config(thread_id)

    try:
        snapshot = approval_agent.get_state(config)

        if not is_waiting_for_approval(snapshot):
            st.error(
                "This workflow is no longer awaiting "
                "approval. No decision was submitted."
            )
            return

        with st.spinner("Processing human decision..."):
            result = approval_agent.invoke(
                Command(
                    resume={
                        "decision": decision,
                        "edited_draft": edited_draft
                    }
                ),
                config=config
            )

        if result.get("__interrupt__"):
            st.warning(
                "The workflow is still waiting "
                "for human input."
            )
            return

        st.session_state.approval_message = (
            result.get(
                "result",
                "Workflow completed."
            )
        )

        clear_pending_draft()
        st.rerun()

    except Exception as error:
        st.error(
            "Approval failed. The checkpoint "
            "may still be recoverable."
        )
        st.code(
            f"{type(error).__name__}: {error}"
        )


# ==========================================
# 7. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Application Info")

    st.write("**AI Model:** Gemma 3 (1B)")
    st.write("**Framework:** LangGraph")
    st.write("**Vector Database:** ChromaDB")
    st.write("**Embeddings:** EmbeddingGemma")
    st.write("**Checkpoints:** SQLite")
    st.write("**Approval Records:** SQLite")

    st.divider()

    st.subheader("Example Questions")

    st.write("What is the Professional plan price?")
    st.write("Check the status of TKT-1001")
    st.write("Draft a reply for TKT-1002")
    st.write("What should I do after HTTP 429?")


# ==========================================
# 8. SMART AGENT
# ==========================================

st.header("💬 Smart AI Agent")

question = st.text_input(
    "Ask your question:",
    placeholder="Draft a reply for TKT-1002",
    key="smart_question"
)

ask_button = st.button(
    "Ask AI Agent",
    type="primary"
)

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Smart agent is working..."):
                result = ask_smart_agent(
                    question.strip()
                )

            intent = result.get("intent", "clarify")
            answer = result.get("answer", "")

            st.session_state.last_intent = intent
            st.session_state.last_answer = answer

            if intent == "draft":

                ticket_id = (
                    result.get("ticket_id")
                    or extract_ticket_id(question)
                )

                if ticket_id:
                    start_draft_workflow(ticket_id)

                else:
                    st.warning(
                        "Please provide a ticket ID."
                    )

        except Exception as error:
            st.error("Smart agent failed.")
            st.code(
                f"{type(error).__name__}: {error}"
            )


# ==========================================
# 9. DISPLAY AGENT RESULT
# ==========================================

if st.session_state.last_answer is not None:

    intent_labels = {
        "docs": "📚 Knowledge Base RAG",
        "ticket": "🔧 Support Ticket Lookup",
        "draft": "✉️ Customer Reply Drafting",
        "clarify": "❓ Clarification",
        "unsupported": "🚫 Out-of-Scope Request"
    }

    label = intent_labels.get(
        st.session_state.last_intent,
        "Smart Agent"
    )

    st.info(f"Workflow: {label}")

    st.subheader("💬 AI Answer")
    st.write(st.session_state.last_answer)


# ==========================================
# 10. DRAFT GENERATION
# ==========================================

st.divider()
st.header("✉️ Customer Reply Drafting")

st.write(
    "Generate a reply, review it, "
    "and approve or reject it."
)

st.caption(
    "Approved replies are saved locally. "
    "No email is automatically sent."
)

ticket_id_input = st.text_input(
    "Ticket ID for reply drafting:",
    placeholder="TKT-1002",
    key="draft_ticket_id"
)

if st.button(
    "Generate Reply Draft",
    disabled=(
        st.session_state.pending_draft is not None
    )
):

    if not ticket_id_input.strip():
        st.warning("Please enter a ticket ID.")

    else:
        start_draft_workflow(ticket_id_input)


# ==========================================
# 11. RECOVER PENDING APPROVAL
# ==========================================

st.divider()
st.header("🔄 Recover Pending Approval")

st.write(
    "If Streamlit restarted while a draft was "
    "awaiting approval, enter its workflow ID "
    "to reopen the saved draft."
)

recovery_id = st.text_input(
    "Saved workflow ID:",
    placeholder="Paste the workflow UUID here",
    key="recovery_thread_id"
)

if st.button(
    "Recover Pending Draft",
    disabled=(
        st.session_state.pending_draft is not None
    )
):
    recover_draft(recovery_id)


# ==========================================
# 12. PENDING APPROVAL
# ==========================================

if st.session_state.pending_draft is not None:

    st.divider()
    st.subheader("📝 Draft Awaiting Approval")

    st.write(
        "**Ticket:** "
        f"{st.session_state.pending_ticket_id}"
    )

    st.write("**Recovery Workflow ID:**")

    # The reviewer must save this ID before
    # restarting the app.
    st.code(
        st.session_state.approval_thread_id,
        language=None
    )

    st.caption(
        "Copy this workflow ID to recover "
        "the draft after restarting Streamlit. "
        "Treat it as sensitive: anyone with "
        "access to this local app and ID may "
        "be able to resume the approval."
    )

    review_key = (
        "reviewed_draft_"
        + st.session_state.approval_thread_id
    )

    edited_draft = st.text_area(
        "Review or edit the reply:",
        value=st.session_state.pending_draft,
        height=280,
        key=review_key
    )

    st.warning(
        "This reply has not been sent. "
        "Verify customer details and company "
        "policies before approving."
    )

    col1, col2 = st.columns(2)

    with col1:
        approve_button = st.button(
            "✅ Approve Draft",
            type="primary"
        )

    with col2:
        reject_button = st.button(
            "❌ Reject Draft"
        )

    if approve_button:
        resume_approval(
            "approve",
            edited_draft
        )

    elif reject_button:
        resume_approval(
            "reject",
            edited_draft
        )


# ==========================================
# 13. APPROVAL RESULT
# ==========================================

if st.session_state.approval_message:

    st.success(
        st.session_state.approval_message
    )

    if st.button("Dismiss message"):
        st.session_state.approval_message = None
        st.rerun()


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(
    "CloudDesk AI Support Copilot | "
    "LangGraph + Ollama + ChromaDB + SQLite"
)
