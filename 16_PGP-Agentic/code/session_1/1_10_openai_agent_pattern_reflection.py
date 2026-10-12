# pip install openai-agents python-dotenv

import asyncio
import json
from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv(override=True)

# -------------------------------------------------
# REFLECTION AGENT
# -------------------------------------------------
reflection_agent = Agent(
    name="CustomerMessageReflectionAgent",
    model="gpt-4o-mini",
    instructions="""
You improve customer-facing incident communication.

Return ONLY valid JSON with exactly these keys:

{
  "draft": "...",
  "reflection": "...",
  "revised_answer": "..."
}

draft:
Write an initial customer-facing message.

reflection:
Critique the draft. Check whether it is:
- clear for non-technical users
- accountable without blaming individuals
- honest about impact
- specific about next steps
- specific about prevention

revised_answer:
Rewrite the message using the reflection.
This should be the final customer-ready version.
"""
)


async def run_reflection(user_request: str):
    result = await Runner.run(reflection_agent, user_request)

    try:
        data = json.loads(result.final_output)
    except json.JSONDecodeError:
        print("Could not parse JSON.")
        print(result.final_output)
        return

    print("\n===== DRAFT =====\n")
    print(data["draft"])

    print("\n===== REFLECTION =====\n")
    print(data["reflection"])

    print("\n===== FINAL REVISED ANSWER =====\n")
    print(data["revised_answer"])


async def main():
    request = """
Write a customer-facing outage update.

Incident:
- Background jobs were delayed for 2 hours
- Some reports and exports were not generated on time
- No customer data was lost
- Root cause was a database connection pool misconfiguration
- The issue is now resolved

Audience:
- Non-technical enterprise customers

Constraints:
- Do not blame individuals
- Avoid internal jargon
- Include what happened, impact, current status, next steps, and prevention
"""

    await run_reflection(request)


if __name__ == "__main__":
    asyncio.run(main())