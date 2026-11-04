
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("ERROR: API key is missing!")
    raise SystemExit(1)

print("API key loaded successfully.")
print("Connecting to OpenAI...")

client = OpenAI(
    api_key=api_key,
    timeout=20.0,
    max_retries=0
)

try:
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Say hello in one sentence."
    )

    print("SUCCESS!")
    print(response.output_text)

except Exception as e:
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR DETAILS:", str(e))
