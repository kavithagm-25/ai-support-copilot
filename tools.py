
import json
from pathlib import Path

TICKETS_FILE = Path(__file__).parent / "data" / "tickets.json"


def lookup_ticket(ticket_id: str) -> dict:
    """Look up a CloudDesk support ticket by its ID."""

    with open(TICKETS_FILE, "r", encoding="utf-8") as file:
        tickets = json.load(file)

    for ticket in tickets:
        if ticket["ticket_id"].upper() == ticket_id.strip().upper():
            return ticket

    return {
        "error": f"Ticket {ticket_id} was not found."
    }
