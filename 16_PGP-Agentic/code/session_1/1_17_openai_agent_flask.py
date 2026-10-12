import os
# pip install flask python-dotenv chromadb openai-agents

from dotenv import load_dotenv
import asyncio

from flask import Flask, request, render_template_string

from agents import Agent, Runner, function_tool

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------
# Setup
# ----------------------------------------------------

load_dotenv(override=True)

app = Flask(__name__)


# ----------------------------------------------------
# ChromaDB
# ----------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "rag", "chromadb")
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="faq_collection",
    embedding_function=embedding_fn
)


# ----------------------------------------------------
# Knowledge Base
# ----------------------------------------------------

knowledge_base = {
    "shipping_time":
        "Our standard shipping time is 3-5 business days.",

    "return_policy":
        "You can return any product within 30 days of delivery.",

    "warranty":
        "All products come with a one-year warranty covering manufacturing defects.",

    "payment_methods":
        "We accept credit cards, debit cards, and PayPal.",

    "customer_support":
        "You can reach our support team 24/7 via email or chat."
}

existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

collection.add(
    documents=list(knowledge_base.values()),
    ids=list(knowledge_base.keys())
)

print("Knowledge base loaded.")


# ----------------------------------------------------
# Tool
# ----------------------------------------------------

@function_tool
async def get_faq_answer(query: str) -> str:

    result = collection.query(
        query_texts=[query],
        n_results=1
    )

    if result["documents"] and result["documents"][0]:
        return result["documents"][0][0]

    return "Sorry, I couldn't find information."


# ----------------------------------------------------
# Agent
# ----------------------------------------------------

faq_agent = Agent(
    name="Customer Support Bot",

    instructions="""
You are a friendly customer support assistant.

Always use the FAQ tool.

If the answer is unavailable,
say that you do not know.

Do not invent answers.
""",

    tools=[get_faq_answer]
)


# ----------------------------------------------------
# Chat Function
# ----------------------------------------------------

async def ask_agent(question):

    result = await Runner.run(
        faq_agent,
        question
    )

    return result.final_output


# ----------------------------------------------------
# HTML
# ----------------------------------------------------

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>Customer Support Bot</title>

<style>

body{
    font-family:Arial;
    width:800px;
    margin:auto;
    margin-top:40px;
}

textarea{
    width:100%;
    height:80px;
}

.answer{
    margin-top:20px;
    padding:15px;
    border:1px solid gray;
    background:#f2f2f2;
}

</style>

</head>

<body>

<h2>Customer Support Bot</h2>

<form method="post">

<textarea
name="question"
placeholder="Ask your question..."
></textarea>

<br><br>

<input
type="submit"
value="Ask">

</form>

{% if question %}

<h3>Your Question</h3>

<p>{{question}}</p>

<h3>Answer</h3>

<div class="answer">

{{answer}}

</div>

{% endif %}

</body>

</html>

"""


# ----------------------------------------------------
# Route
# ----------------------------------------------------

@app.route("/", methods=["GET", "POST"])

def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        answer = asyncio.run(
            ask_agent(question)
        )

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )


# ----------------------------------------------------
# Main
# ----------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)