# pip install openai-agents python-dotenv requests

import asyncio
import requests
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool

load_dotenv(override=True)

# -------------------------------------------------
# Simple real tools
# -------------------------------------------------
@function_tool
def get_weather(city: str) -> str:
    """Get current weather for a city using wttr.in."""
    url = f"https://wttr.in/{city}?format=3"

    try:
        return requests.get(url, timeout=10).text
    except Exception as e:
        return f"Could not fetch weather: {e}"


@function_tool
def get_air_quality(city: str) -> str:
    """Get current air quality for a city using Open-Meteo."""
    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        geo_data = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        ).json()

        results = geo_data.get("results", [])
        if not results:
            return f"Could not find location for city: {city}"

        location = results[0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_name = location["name"]
        country = location.get("country", "")

        aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        aq_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "us_aqi,pm2_5,pm10",
            "timezone": "auto"
        }

        aq_data = requests.get(
            aq_url,
            params=aq_params,
            timeout=10
        ).json()

        current = aq_data.get("current", {})
        aqi = current.get("us_aqi")
        pm25 = current.get("pm2_5")
        pm10 = current.get("pm10")

        if aqi is None:
            return f"Air quality data not available for {resolved_name}, {country}"

        if aqi <= 50:
            category = "Good"
        elif aqi <= 100:
            category = "Moderate"
        elif aqi <= 150:
            category = "Unhealthy for sensitive groups"
        elif aqi <= 200:
            category = "Unhealthy"
        elif aqi <= 300:
            category = "Very unhealthy"
        else:
            category = "Hazardous"

        return (
            f"Air quality for {resolved_name}, {country}: "
            f"US AQI {aqi} ({category}). "
            f"PM2.5: {pm25}, PM10: {pm10}."
        )

    except Exception as e:
        return f"Could not fetch air quality: {e}"


@function_tool
def get_walking_safety_rules() -> str:
    """Return basic safety rules for deciding whether a walk is safe."""
    return """
Walking is usually safe if:
- Weather is not too hot, stormy, or rainy
- Air quality is good or moderate
- The user avoids isolated areas late at night
- The user carries water for long walks
"""


# -------------------------------------------------
# ReAct Agent
# -------------------------------------------------
walking_agent = Agent(
    name="WalkingAdvisor",
    model="gpt-4o-mini",
    tools=[
        get_weather,
        get_air_quality,
        get_walking_safety_rules,
    ],
    instructions="""
You are a simple ReAct-style walking advisor.

Before answering, use tools to check:
1. Weather
2. Air quality
3. Walking safety rules

Then give a short final recommendation:
- Yes, go for a walk
- Go, but take precautions
- Better to avoid it

Explain your reason briefly.
"""
)


# -------------------------------------------------
# Run
# -------------------------------------------------
async def main():
    user_question = """
Can I go for a 45-minute walk in Pune today?
"""

    result = await Runner.run(walking_agent, user_question)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())