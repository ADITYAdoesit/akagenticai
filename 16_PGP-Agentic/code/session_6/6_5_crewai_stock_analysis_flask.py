# pip install flask requests python-dotenv crewai

from flask import Flask, request, render_template_string
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai.tools import tool
import os

load_dotenv(override=True)

app = Flask(__name__)

MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY")

# =========================
# TOOL: REAL MARKET NEWS
# =========================

@tool
def fetch_market_news(stock_symbol: str) -> str:
    """
    Fetch real, recent market news for a stock symbol using MarketAux API.
    """
    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": stock_symbol,
        "language": "en",
        "limit": 5,
        "api_token": MARKETAUX_API_KEY
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json().get("data", [])

    if not data:
        return "No recent news found."

    news_text = []
    for item in data:
        news_text.append(
            f"- {item.get('title')} ({item.get('source')})"
        )

    return "\n".join(news_text)

# =========================
# AGENTS
# =========================

news_agent = Agent(
    role="Market News Analyst",
    goal="Analyze stock using real market news",
    backstory="You analyze only provided news data. You never invent events.",
    tools=[fetch_market_news],
    llm="gpt-4o-mini",
    verbose=True
)

financial_agent = Agent(
    role="Financial Analyst",
    goal="Provide high-level financial reasoning without fabricating numbers",
    backstory="You reason qualitatively unless verified data is provided.",
    llm="gpt-4o-mini",
    verbose=True
)

# =========================
# MAIN FUNCTION
# =========================

def analyze_stock(stock_symbol: str):
    if not stock_symbol:
        return "Please enter a stock symbol."

    # ---- Tasks ----

    news_task = Task(
        description=(
            f"Use the tool to fetch REAL news for {stock_symbol}.\n"
            "Summarize sentiment based ONLY on that news.\n"
            "Do not add external facts."
        ),
        expected_output="News-based sentiment analysis grounded in fetched headlines",
        agent=news_agent
    )

    finance_task = Task(
        description=(
            f"Provide a QUALITATIVE financial perspective on {stock_symbol}.\n"
            "If numbers are unknown, explicitly say so."
        ),
        expected_output="High-level financial reasoning with uncertainty",
        agent=financial_agent
    )

    crew = Crew(
        agents=[news_agent, financial_agent],
        tasks=[news_task, finance_task],
        verbose=True
    )

    return str(crew.kickoff())

# --------------------------------------------------
# HTML
# --------------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Stock Analysis</title>
<style>
body{font-family:Arial;width:900px;margin:auto;margin-top:40px}
.answer{margin-top:20px;padding:15px;border:1px solid #999;background:#f2f2f2;white-space:pre-wrap}
</style>
</head>
<body>

<h2>Stock Analysis</h2>
<p>Uses real market news via MarketAux combined with qualitative analysis.</p>

<form method="post">

<label for="question">Stock Symbol</label><br>
<input
type="text"
id="question"
name="question"
value="{{question or 'INFY'}}"
placeholder="e.g. INFY">

<br><br>

<input type="submit" value="Analyze">

</form>

{% if answer %}

<h3>Analysis</h3>

<div class="answer">
{{answer}}
</div>

{% endif %}

</body>
</html>
"""

# --------------------------------------------------
# Flask Route
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        answer = analyze_stock(question)

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
