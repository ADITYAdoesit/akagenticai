# pip install pydantic autogen-agentchat autogen-ext python-dotenv requests

import asyncio
import requests
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent

load_dotenv(override=True)

# Pydantic models for structured output
class EvidenceItem(BaseModel):
    country: str
    years: List[int]
    yearly_growth: List[Optional[float]]
    note: Optional[str] = None

class FactCheckReport(BaseModel):
    claim: str
    verdict: str
    summary: str
    evidence: List[EvidenceItem]
    confidence: str


def fetch_gdp_growth(country_code: str, start_year: int = 1995, end_year: int = 2024):
    """Fetch annual GDP growth data from World Bank API."""
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.KD.ZG?format=json&date={start_year}:{end_year}&per_page=1000"
    data = requests.get(url, timeout=20).json()
    if len(data) < 2:
        # API sometimes returns metadata only, so handle missing data
        return [], []
    
    # Convert JSON response to a dictionary of {year: growth_value}
    series = {
        int(item["date"]): item["value"]
        for item in data[1]
        if item.get("date") and item.get("value") is not None
    }
    
    # Sort years to maintain chronological order
    years = sorted(series.keys())
    growths = [series[y] for y in years]
    return years, growths # both will be lists


def detect_recession(years: List[int], growths: List[float]):
    """Detect two consecutive years of negative GDP growth (annual proxy for recession)."""
    recessions = []
    for i in range(len(years) - 1):
        # Check if both consecutive years had negative growth
        if growths[i] < 0 and growths[i + 1] < 0:
            recessions.append((years[i], years[i + 1]))
    return recessions


async def check_recession(country_code: str, country_name: str) -> EvidenceItem:
    """Run GDP fetch + recession detection for a given country asynchronously."""
    # Run blocking API call in executor to avoid blocking event loop
    years, growths = await asyncio.get_event_loop().run_in_executor(None, fetch_gdp_growth, country_code)
    
    # Detect recession years based on consecutive negative growth
    recessions = detect_recession(years, growths)
    
    # Summarize findings for this country
    note = f"Recessions found: {recessions}" if recessions else "No consecutive negative growth found"
    
    return EvidenceItem(
        country=country_name,
        years=years,
        yearly_growth=[round(g, 2) for g in growths],
        note=note
    )


# Define fact-checking agent (for integration or expansion later)
agent = AssistantAgent(
    name="fact_checker",
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    tools=[check_recession],
    system_message="You are a fact-checker. Use check_recession tool to verify claims about economic recessions.",
)


async def main():
    countries = [
        ("IN", "India"),
        ("US", "United States"),
        ("GB", "United Kingdom"),
        ("JP", "Japan"),
    ]
    
    # Collect recession evidence for all countries
    evidence = []
    for code, name in countries:
        evidence.append(await check_recession(code, name))
    
    # Decide verdict based on India's evidence
    india_note = next((e.note for e in evidence if "India" in e.country), "")
    verdict = (
        "Not supported"
        if "Recessions found:" in india_note and "No consecutive" not in india_note
        else "Supported"
    )
    
    # Build structured report summarizing results
    report = FactCheckReport(
        claim="India has never faced a recession in last 30 years",
        verdict=verdict,
        summary=f"Analysis based on annual GDP growth data. {verdict.lower()} by evidence.",
        evidence=evidence,
        confidence="Medium",
    )
    
    # Print JSON report neatly
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
