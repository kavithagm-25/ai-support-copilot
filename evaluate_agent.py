
import csv
from datetime import datetime
from pathlib import Path

from smart_agent import router_node, ask_smart_agent


# ==========================================
# 1. UNSEEN ROUTING TEST DATA
# ==========================================

# These questions are different from the
# examples in our classifier prompt.

TEST_CASES = [
    {
        "question": "How much does CloudDesk Basic cost monthly?",
        "expected": "docs"
    },
    {
        "question": "Can I cancel my CloudDesk subscription?",
        "expected": "docs"
    },
    {
        "question": "How many API requests can I make per minute?",
        "expected": "docs"
    },
    {
        "question": "Where can I enable two-factor authentication?",
        "expected": "docs"
    },
    {
        "question": "Is TKT-1003 already resolved?",
        "expected": "ticket"
    },
    {
        "question": "Tell me the priority of TKT-1001",
        "expected": "ticket"
    },
    {
        "question": "Find the details for TKT-1002",
        "expected": "ticket"
    },
    {
        "question": "Prepare an email for the customer on TKT-1002",
        "expected": "draft"
    },
    {
        "question": "Help me respond to the customer for TKT-1001",
        "expected": "draft"
    },
    {
        "question": "Compose a support message about TKT-1003",
        "expected": "draft"
    },
    {
        "question": "Can you look into my ticket?",
        "expected": "clarify"
    },
    {
        "question": "Please write a reply for my support case",
        "expected": "clarify"
    },
    {
        "question": "Tell me about it",
        "expected": "clarify"
    },
    {
        "question": "Give me a recipe for pasta",
        "expected": "unsupported"
    },
    {
        "question": "What is the capital of Australia?",
        "expected": "unsupported"
    },
    {
        "question": "Recommend a movie to watch tonight",
        "expected": "unsupported"
    }
]


# ==========================================
# 2. EVALUATE ROUTING
# ==========================================

def evaluate_routing():

    results = []
    passed = 0

    print("\nStarting routing evaluation...\n")

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected = test["expected"]

        try:
            # Test only the router, not the
            # slower RAG answer-generation step.
            result = router_node({
                "question": question
            })

            actual = result["intent"]
            success = actual == expected
            error = ""

        except Exception as exc:
            actual = "error"
            success = False
            error = str(exc)

        if success:
            passed += 1

        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "passed": success,
            "error": error
        })

        status = "PASS" if success else "FAIL"

        print(
            f"{index:02d}. {status} | "
            f"Expected: {expected} | Actual: {actual}"
        )

        if not success:
            print(f"    Question: {question}")
            if error:
                print(f"    Error: {error}")

    total = len(TEST_CASES)
    accuracy = passed / total * 100

    print("\n" + "=" * 55)
    print(f"ROUTING ACCURACY: {accuracy:.1f}%")
    print(f"PASSED: {passed}/{total}")
    print("=" * 55)

    return results


# ==========================================
# 3. CHECK TICKET AND APPROVAL SAFETY
# ==========================================

def evaluate_safety():

    checks = []

    # A missing ticket ID must not allow
    # ticket lookup or reply drafting.
    for question in [
        "Check my support ticket",
        "Draft a reply for my ticket"
    ]:

        result = router_node({
            "question": question
        })

        checks.append({
            "test": question,
            "passed": result["intent"] not in (
                "ticket", "draft"
            )
        })

    # A nonexistent ticket must not be
    # treated as a valid drafting request.
    result = ask_smart_agent(
        "Draft a reply for TKT-9999"
    )

    checks.append({
        "test": "Missing ticket blocks drafting",
        "passed": (
            "not found" in result["answer"].lower()
        )
    })

    print("\nSAFETY CHECKS")

    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status}: {check['test']}")

    return checks


# ==========================================
# 4. SAVE EVALUATION RESULTS
# ==========================================

def save_results(results):

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        output_dir / f"routing_eval_{timestamp}.csv"
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "expected",
                "actual",
                "passed",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


# ==========================================
# 5. RUN EVALUATION
# ==========================================

if __name__ == "__main__":

    routing_results = evaluate_routing()
    safety_results = evaluate_safety()

    save_results(routing_results)

    safety_passed = sum(
        item["passed"] for item in safety_results
    )

    print(
        f"\nSAFETY SCORE: "
        f"{safety_passed}/{len(safety_results)}"
    )
