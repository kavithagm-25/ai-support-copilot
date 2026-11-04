
from smart_agent import ask_smart_agent


test_cases = [
    (
        "How do I reset my password?",
        "docs"
    ),
    (
        "What is the Professional plan price?",
        "docs"
    ),
    (
        "What's happening with TKT-1001?",
        "ticket"
    ),
    (
        "Check the status of TKT-1002",
        "ticket"
    ),
    (
        "Draft a reply for TKT-1002",
        "draft"
    ),
    (
        "Write a customer response for TKT-1001",
        "draft"
    ),
    (
        "Check my support ticket",
        "clarify"
    ),
    (
        "What's the weather today?",
        "unsupported"
    )
]


passed = 0

for question, expected_intent in test_cases:

    print("\n" + "=" * 55)
    print("QUESTION:", question)

    try:
        result = ask_smart_agent(question)

        actual_intent = result["intent"]

        print("EXPECTED:", expected_intent)
        print("ACTUAL:", actual_intent)
        print("ANSWER:", result["answer"])

        if actual_intent == expected_intent:
            print("RESULT: PASS")
            passed += 1
        else:
            print("RESULT: FAIL")

    except Exception as error:
        print("RESULT: ERROR")
        print(type(error).__name__, str(error))


print("\n" + "=" * 55)
print(f"FINAL SCORE: {passed}/{len(test_cases)}")
