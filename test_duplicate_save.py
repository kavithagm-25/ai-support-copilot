
import sqlite3
from uuid import uuid4

from approval_agent import (
    RECORDS_DB,
    save_approved_record,
    export_approved_drafts
)


def main():

    print("\nDUPLICATE SAVE PROTECTION TEST\n")

    approval_id = str(uuid4())

    ticket_id = "TKT-1002"

    reply = (
        "Hello, thank you for contacting "
        "CloudDesk support. Our team will "
        "review your request."
    )

    print("TEST 1: First save")

    first = save_approved_record(
        approval_id,
        ticket_id,
        reply
    )

    print("Inserted:", first)

    print("\nTEST 2: Retry same approval")

    second = save_approved_record(
        approval_id,
        ticket_id,
        reply
    )

    print("Inserted:", second)

    print("\nTEST 3: Check SQLite count")

    with sqlite3.connect(
        str(RECORDS_DB)
    ) as connection:

        count = connection.execute(
            """
            SELECT COUNT(*)
            FROM approved_replies
            WHERE approval_id = ?
            """,
            (approval_id,)
        ).fetchone()[0]

    print("Matching records:", count)

    print("\nTEST 4: Refresh JSONL export")

    export_approved_drafts()

    print("Export completed.")

    print("\n" + "=" * 50)

    passed = (
        first is True
        and second is False
        and count == 1
    )

    if passed:
        print(
            "FINAL RESULT: "
            "DUPLICATE PROTECTION PASSED"
        )
    else:
        print(
            "FINAL RESULT: "
            "DUPLICATE PROTECTION FAILED"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()
