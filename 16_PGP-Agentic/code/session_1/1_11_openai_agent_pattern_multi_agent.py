import os
import asyncio
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

# --------------------------------
# UTILITY: Clean JSON output from agent
# --------------------------------
def extract_json(text: str) -> str:
    """
    Extract the first {...} block from text.
    Useful when the model accidentally adds extra text.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def safe_parse_json(text: str) -> dict:
    """
    Parse JSON safely, with fallback to raw text.
    """
    try:
        clean_text = extract_json(text)
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {
            "error": "Invalid JSON",
            "raw_output": text
        }

# --------------------------------
# AGENT 1: EXTRACTOR
# --------------------------------
extractor_agent = Agent(
    name="CandidateExtractorAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Candidate Extractor Agent for HR screening.

Task:
- Extract and structure candidate details
- Preserve factual information from the input
- Do not make a hiring decision

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include any extra explanation or text.
- Escape newlines inside strings.

JSON FORMAT:
{
  "Candidate": {
    "name": "...",
    "skills": [...],
    "experience_years": 0,
    "projects": [...],
    "statement": "..."
  },
  "Role": "Agentic AI Developer/Architect"
}
"""
)

# --------------------------------
# AGENT 2: EVALUATOR
# --------------------------------
evaluator_agent = Agent(
    name="CandidateEvaluatorAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Candidate Evaluator Agent for HR screening.

Input:
- Candidate Extractor Agent JSON output

Task:
- Compare candidate skills, experience, projects, and statement to the role
- Assess suitability for Agentic AI Developer/Architect
- Do not make the final interview/reject decision

Evaluation criteria:
- Agentic AI experience
- LLM application development
- Python experience
- Multi-agent or workflow orchestration experience
- Architecture/design ability
- Relevant project evidence

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include extra text or commentary.

JSON FORMAT:
{
  "Suitability": {
    "status": "strong" | "moderate" | "weak",
    "reason": "..."
  },
  "Strengths": [...],
  "Risks": [...]
}
"""
)

# --------------------------------
# AGENT 3: DECIDER
# --------------------------------
decider_agent = Agent(
    name="CandidateDecisionAgent",
    model="gpt-4o-mini",
    instructions="""
You are a Candidate Decision Agent for HR screening.

Input:
- Extractor Agent JSON output
- Evaluator Agent JSON output

Task:
- Decide whether to invite candidate for interview or reject
- Provide final justification

Decision rules:
- If suitability is "strong", invite for interview.
- If suitability is "moderate", invite only if strengths outweigh risks.
- If suitability is "weak", reject.

VERY IMPORTANT:
- Output MUST be valid JSON only.
- Do NOT include extra explanation or commentary.

JSON FORMAT:
{
  "Candidate": {...},
  "Role": "Agentic AI Developer/Architect",
  "Suitability": {...},
  "Decision": "invite_for_interview" | "reject",
  "Justification": "..."
}
"""
)

# --------------------------------
# PROCESS SINGLE CANDIDATE
# --------------------------------
async def process_candidate(candidate: dict) -> dict:
    extractor_input = f"""
Candidate Details:
- Name: {candidate['name']}
- Skills: {', '.join(candidate['skills'])}
- Experience (years): {candidate['experience_years']}
- Projects: {', '.join(candidate['projects'])}
- Statement: {candidate['statement']}

Role: Agentic AI Developer/Architect
"""

    # Step 1: Extract candidate profile
    extractor_result = await Runner.run(extractor_agent, extractor_input)
    extractor_json = safe_parse_json(extractor_result.final_output)

    # Step 2: Evaluate candidate
    evaluator_result = await Runner.run(
        evaluator_agent,
        json.dumps(extractor_json, indent=2)
    )
    evaluator_json = safe_parse_json(evaluator_result.final_output)

    # Step 3: Make final decision
    decider_input = json.dumps(
        {
            "ExtractorOutput": extractor_json,
            "EvaluatorOutput": evaluator_json
        },
        indent=2
    )

    decision_result = await Runner.run(decider_agent, decider_input)
    decision_json = safe_parse_json(decision_result.final_output)

    return decision_json

# --------------------------------
# MAIN LOOP
# --------------------------------
async def main():
    input_file = Path(os.path.join(BASE_DIR, "candidates.json"))
    output_file = Path(os.path.join(BASE_DIR, "candidate_decisions.json"))

    with input_file.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    all_decisions = []

    for candidate in candidates:
        decision_json = await process_candidate(candidate)
        all_decisions.append(decision_json)
        print(f"Processed candidate: {candidate['name']}")

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_decisions, f, indent=2)

    print(f"\nAll candidate decisions written to {output_file.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())