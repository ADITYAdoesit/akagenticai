# pip install openai-agents python-dotenv requests

import asyncio
import json
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool

load_dotenv(override=True)


# -------------------------------------------------
# Helper
# -------------------------------------------------
def get_location(city: str):
    geo = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        },
        timeout=10,
    ).json()

    results = geo.get("results", [])
    if not results:
        return None

    return results[0]


# -------------------------------------------------
# REAL TOOLS
# -------------------------------------------------
@function_tool
def get_current_weather(city: str) -> str:
    """Get current weather for a city using Open-Meteo."""
    loc = get_location(city)
    if not loc:
        return f"Location not found: {city}"

    weather = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    current = weather.get("current", {})

    return (
        f"Weather in {loc['name']}, {loc.get('country', '')}: "
        f"temperature={current.get('temperature_2m')}C, "
        f"humidity={current.get('relative_humidity_2m')}%, "
        f"precipitation={current.get('precipitation')}mm, "
        f"wind_speed={current.get('wind_speed_10m')}km/h"
    )


@function_tool
def get_air_quality(city: str) -> str:
    """Get current air quality for a city using Open-Meteo."""
    loc = get_location(city)
    if not loc:
        return f"Location not found: {city}"

    aq = requests.get(
        "https://air-quality-api.open-meteo.com/v1/air-quality",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "us_aqi,pm2_5,pm10",
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    current = aq.get("current", {})
    aqi = current.get("us_aqi")

    if aqi is None:
        return f"Air quality data unavailable for {loc['name']}"

    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Moderate"
    elif aqi <= 150:
        category = "Unhealthy for sensitive groups"
    elif aqi <= 200:
        category = "Unhealthy"
    else:
        category = "Very unhealthy or hazardous"

    return (
        f"Air quality in {loc['name']}, {loc.get('country', '')}: "
        f"US AQI={aqi} ({category}), "
        f"PM2.5={current.get('pm2_5')}, "
        f"PM10={current.get('pm10')}"
    )


@function_tool
def get_uv_index(city: str) -> str:
    """Get current UV index for a city using Open-Meteo."""
    loc = get_location(city)
    if not loc:
        return f"Location not found: {city}"

    uv = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "current": "uv_index",
            "timezone": "auto",
        },
        timeout=10,
    ).json()

    current = uv.get("current", {})
    uv_index = current.get("uv_index")

    if uv_index is None:
        return f"UV index data unavailable for {loc['name']}"

    if uv_index < 3:
        risk = "Low"
    elif uv_index < 6:
        risk = "Moderate"
    elif uv_index < 8:
        risk = "High"
    elif uv_index < 11:
        risk = "Very high"
    else:
        risk = "Extreme"

    return (
        f"UV index in {loc['name']}, {loc.get('country', '')}: "
        f"{uv_index} ({risk} risk)"
    )


@function_tool
def get_activity_rules(activity: str) -> str:
    """Return simple safety rules for outdoor activity."""
    return (
        f"For {activity}: avoid outdoor exercise if AQI is above 150, "
        "avoid heavy activity in extreme heat, rain, or strong wind, "
        "use sun protection when UV index is high, "
        "and prefer morning/evening if temperature or UV is high."
    )


# -------------------------------------------------
# PLANNER
# -------------------------------------------------
planner_agent = Agent(
    name="OutdoorActivityPlanner",
    model="gpt-4o-mini",
    instructions="""
You are a planning agent.

Create a short execution plan for deciding whether the user's outdoor activity is safe.

Return ONLY a JSON array of strings.
Use exactly 4 steps:
1. Check current weather.
2. Check current air quality.
3. Check current UV index.
4. Check activity-specific safety rules.

Do not execute anything.
""",
)


# -------------------------------------------------
# EXECUTOR
# -------------------------------------------------
executor_agent = Agent(
    name="OutdoorActivityExecutor",
    model="gpt-4o-mini",
    tools=[
        get_current_weather,
        get_air_quality,
        get_uv_index,
        get_activity_rules,
    ],
    instructions="""
You execute exactly one step from a plan.

Use tools when the step requires real information.
Report the actual tool result.
Do not jump ahead.
""",
)


# -------------------------------------------------
# FINAL SYNTHESIZER
# -------------------------------------------------
final_agent = Agent(
    name="OutdoorActivityFinalAdvisor",
    model="gpt-4o-mini",
    instructions="""
Use the collected findings to give a practical final recommendation.

Answer format:
- Recommendation
- Evidence
- Precautions
""",
)


async def run_planning_pattern(user_query: str):
    plan_result = await Runner.run(planner_agent, user_query)

    try:
        steps = json.loads(plan_result.final_output)
    except json.JSONDecodeError:
        steps = [plan_result.final_output]

    print("\n===== PLAN =====")
    for i, step in enumerate(steps, 1):
        print(f"{i}. {step}")

    findings = []

    for i, step in enumerate(steps, 1):
        executor_input = f"""
User request:
{user_query}

Full plan:
{json.dumps(steps, indent=2)}

Findings so far:
{json.dumps(findings, indent=2)}

Current step:
{step}
"""
        result = await Runner.run(executor_agent, executor_input)

        findings.append({
            "step": step,
            "result": result.final_output,
        })

        print(f"\n===== STEP {i} RESULT =====")
        print(result.final_output)

    final_input = f"""
User request:
{user_query}

Findings:
{json.dumps(findings, indent=2)}

Give final recommendation.
"""
    final_result = await Runner.run(final_agent, final_input)

    print("\n===== FINAL RECOMMENDATION =====")
    print(final_result.final_output)


async def main():
    user_query = "Can I go for a 45-minute walk in Pune today?"
    await run_planning_pattern(user_query)


if __name__ == "__main__":
    asyncio.run(main())
