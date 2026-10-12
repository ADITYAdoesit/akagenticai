import os
import asyncio
import json
from dotenv import load_dotenv
from agents import Agent, Runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

# --------------------------------
# WORKER TOOL: deterministic code
# --------------------------------
def compute_eligibility(marks: float, interview_score: float, cutoff_marks: float) -> dict:
    average = (marks + interview_score) / 2

    if average >= cutoff_marks:
        status = "eligible"
        reason = "Average score meets or exceeds the cutoff."
    elif average >= cutoff_marks * 0.95:
        status = "borderline"
        reason = "Average score is within 5% of the cutoff."
    else:
        status = "not eligible"
        reason = "Average score is below the acceptable cutoff range."

    return {
        "average_score": average,
        "cutoff_marks": cutoff_marks,
        "status": status,
        "reason": reason
    }


def assess_statement(statement: str) -> dict:
    statement_lower = statement.lower()

    strong_signals = [
        "research",
        "project",
        "leadership",
        "internship",
        "community",
        "achievement",
        "scholarship",
        "competition",
        "volunteer"
    ]

    matched = [
        signal for signal in strong_signals
        if signal in statement_lower
    ]

    if len(matched) >= 2:
        strength = "strong"
    elif len(matched) == 1:
        strength = "moderate"
    else:
        strength = "weak"

    return {
        "statement_strength": strength,
        "matched_signals": matched,
        "reason": f"Statement contains {len(matched)} positive admission signal(s)."
    }


# --------------------------------
# PLANNER: creates evidence plan only
# --------------------------------
planner_agent = Agent(
    name="AdmissionPlannerREWOO",
    model="gpt-4o-mini",
    instructions="""
You are the Planner in a ReWOO pipeline.

ReWOO means:
- Plan first
- Do not observe tool results while planning
- Worker executes the planned steps later
- Solver uses the worker evidence to answer

Given applicant data, output ONLY valid JSON:

{
  "plan": [
    {
      "id": "E1",
      "tool": "compute_eligibility",
      "args": {
        "marks": 0,
        "interview_score": 0,
        "cutoff_marks": 0
      }
    },
    {
      "id": "E2",
      "tool": "assess_statement",
      "args": {
        "statement": "..."
      }
    }
  ]
}

Do not compute eligibility yourself.
Do not make the final decision.
"""
)

# --------------------------------
# SOLVER: final decision from evidence
# --------------------------------
solver_agent = Agent(
    name="AdmissionSolverREWOO",
    model="gpt-4o-mini",
    instructions="""
You are the Solver in a ReWOO pipeline.

Use only the provided worker evidence.
Do not recompute scores.

Decision rules:
- If eligibility status is "eligible", admit.
- If eligibility status is "borderline" and statement strength is "strong", admit.
- If eligibility status is "borderline" and statement strength is not strong, reject.
- If eligibility status is "not eligible", reject.

Output ONLY valid JSON:

{
  "Decision": "admit" | "reject",
  "Eligibility": {
    "status": "...",
    "reason": "..."
  },
  "StatementReview": {
    "strength": "...",
    "reason": "..."
  },
  "Justification": "..."
}
"""
)


# --------------------------------
# WORKER EXECUTION
# --------------------------------
def execute_worker_step(step: dict) -> dict:
    tool = step["tool"]
    args = step["args"]

    if tool == "compute_eligibility":
        result = compute_eligibility(**args)
    elif tool == "assess_statement":
        result = assess_statement(**args)
    else:
        result = {"error": f"Unknown tool: {tool}"}

    return {
        "id": step["id"],
        "tool": tool,
        "result": result
    }


# --------------------------------
# PROCESS EACH STUDENT
# --------------------------------
async def process_student(student):
    plan_query = f"""
Applicant:
{json.dumps(student, indent=2)}

Create a ReWOO evidence plan.
"""

    # 1. Plan
    plan_result = await Runner.run(planner_agent, plan_query)

    try:
        plan_data = json.loads(plan_result.final_output)
        plan = plan_data["plan"]
    except (json.JSONDecodeError, KeyError):
        print(f"\nERROR parsing plan for {student['name']}")
        print(plan_result.final_output)
        return

    # 2. Work
    evidence = []
    for step in plan:
        evidence.append(execute_worker_step(step))

    # 3. Solve
    solve_query = f"""
Applicant name:
{student["name"]}

Worker evidence:
{json.dumps(evidence, indent=2)}
"""

    result = await Runner.run(solver_agent, solve_query)

    try:
        data = json.loads(result.final_output)

        print(f"\n=== Student: {student['name']} ===")
        print("Decision:", data.get("Decision"))
        print("Eligibility:", data.get("Eligibility", {}).get("status"))
        print("Eligibility Reason:", data.get("Eligibility", {}).get("reason"))
        print("Statement Strength:", data.get("StatementReview", {}).get("strength"))
        print("Justification:", data.get("Justification"))

    except json.JSONDecodeError:
        print(f"\nERROR parsing solver JSON for {student['name']}")
        print(result.final_output)


# --------------------------------
# MAIN LOOP
# --------------------------------
async def main():
    with open(os.path.join(BASE_DIR, "students.json"), "r") as f:
        students = json.load(f)

    for student in students:
        await process_student(student)


if __name__ == "__main__":
    asyncio.run(main())