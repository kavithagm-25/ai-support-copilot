
import sys

from langgraph.types import Command
from approval_agent import approval_agent


THREAD_ID = "persistence-test-tkt-1002"

config = {
    "configurable": {
        "thread_id": THREAD_ID
    }
}


def show_status():
    snapshot = approval_agent.get_state(config)

    if not snapshot.values:
        print("No saved workflow found.")
        return

    print("Saved ticket:", snapshot.values.get("ticket_id"))
    print("Next node:", snapshot.next)

    if snapshot.next:
        print("Workflow is paused.")
        print(
            "Draft:",
            snapshot.values.get("draft", "")[:300]
        )
    else:
        print(
            "Workflow completed:",
            snapshot.values.get("result")
        )


if len(sys.argv) != 2:
    print(
        "Usage: python test_persistence.py "
        "start|status|reject"
    )
    sys.exit(1)


command = sys.argv[1].lower()


if command == "start":

    # Avoid overwriting an existing pending test.
    snapshot = approval_agent.get_state(config)

    if snapshot.next:
        print("A pending test already exists.")
        show_status()
        sys.exit(0)

    result = approval_agent.invoke(
        {"ticket_id": "TKT-1002"},
        config=config
    )

    if result.get("__interrupt__"):
        print("SUCCESS: Workflow paused for approval.")
    else:
        print("Unexpected result:", result)


elif command == "status":
    show_status()


elif command == "reject":

    snapshot = approval_agent.get_state(config)

    if not snapshot.next:
        print("No pending workflow to reject.")
        sys.exit(0)

    result = approval_agent.invoke(
        Command(
            resume={"decision": "reject"}
        ),
        config=config
    )

    print("Result:", result.get("result"))


else:
    print("Unknown command.")
