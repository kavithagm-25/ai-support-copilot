
import csv
from datetime import datetime
from pathlib import Path

from smart_agent import ask_smart_agent


TEST_CASES = [
    {
        "question": "How long is a password reset link valid?",
        "expected_terms": ["30", "minute"]
    },
    {
        "question": "What is the monthly price of the Professional plan?",
        "expected_terms": ["25"]
    },
    {
        "question": "How many days do customers have to request a refund?",
        "expected_terms": ["14"]
    },
    {
        "question": "What should I do if I receive HTTP 429?",
        "expected_terms": ["wait", "retry"]
    },
    {
        "question": "What is the API request limit per minute?",
        "expected_terms": ["100"]
    }
]


def evaluate_rag():

    results = []
    passed = 0

    print("\nStarting RAG answer evaluation...\n")

    for index, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected_terms = test["expected_terms"]

        try:
            result = ask_smart_agent(question)

            answer = result.get("answer", "")
            intent = result.get("intent", "")

            answer_lower = answer.lower()

            missing_terms = [
                term for term in expected_terms
                if term.lower() not in answer_lower
            ]

            success = (
                intent == "docs"
                and not missing_terms
            )

            error = ""

        except Exception as exc:
            answer = ""
            intent = "error"
            missing_terms = expected_terms
            success = False
            error = str(exc)

        if success:
            passed += 1

        results.append({
            "question": question,
            "intent": intent,
            "expected_terms": ", ".join(expected_terms),
            "answer": answer,
            "passed": success,
            "error": error
        })

        status = "PASS" if success else "FAIL"

        print("=" * 55)
        print(f"TEST {index}: {status}")
        print(f"Question: {question}")
        print(f"Intent: {intent}")
        print(f"Answer: {answer}")

        if missing_terms:
            print(
                "Missing expected terms: "
                + ", ".join(missing_terms)
            )

        if error:
            print(f"Error: {error}")

    total = len(TEST_CASES)
    accuracy = passed / total * 100

    print("\n" + "=" * 55)
    print(f"RAG FACT CHECK SCORE: {passed}/{total}")
    print(f"RAG FACT MATCH RATE: {accuracy:.1f}%")
    print("=" * 55)

    return results


def save_results(results):

    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = (
        output_dir / f"rag_eval_{timestamp}.csv"
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
                "intent",
                "expected_terms",
                "answer",
                "passed",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":

    results = evaluate_rag()
    save_results(results)
