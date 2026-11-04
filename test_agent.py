
from agent import ask_agent

questions = [
    "What is the status of ticket TKT-1001?",
    "What is the status of ticket TKT-9999?",
    "How long is the password reset link valid?",
    "What should I do if the API returns HTTP 429?"
]

for question in questions:
    print("\n" + "=" * 50)
    print("QUESTION:", question)
    print("ANSWER:", ask_agent(question))
