
from smart_agent import ask_smart_agent


QUESTIONS = [
    "Who is the CEO of CloudDesk?",
    "Where is CloudDesk headquarters located?",
    "What is CloudDesk's guaranteed uptime percentage?",
    "Does CloudDesk offer a lifetime subscription?"
]


def main():
    print("\nCLOUDDESK HALLUCINATION TEST\n")

    for number, question in enumerate(QUESTIONS, start=1):
        print("=" * 60)
        print(f"TEST {number}")
        print(f"QUESTION: {question}")

        try:
            result = ask_smart_agent(question)

            print(f"INTENT: {result.get('intent')}")
            print(f"ANSWER: {result.get('answer')}")

        except Exception as error:
            print(f"ERROR: {error}")

    print("\n" + "=" * 60)
    print("Review each answer for unsupported claims.")
    print("The agent should acknowledge missing information.")
    print("This test does not automatically score correctness.")


if __name__ == "__main__":
    main()
