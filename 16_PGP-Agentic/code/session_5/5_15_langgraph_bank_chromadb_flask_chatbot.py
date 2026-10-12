import os
# pip install flask langchain-community langgraph pandas chromadb

from flask import Flask, request, render_template_string
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
import pandas as pd
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# --------------------------
# State
# --------------------------

class ChatState(TypedDict):
    query: str
    answer: str

# --------------------------
# Load dataset
# --------------------------

csv_path = os.path.join(BASE_DIR, "Dataset_banking_chatbot.csv")

df = pd.read_csv(csv_path, encoding="latin-1")

texts = df["Query"].astype(str).tolist()
metadatas = df[["Query", "Response"]].to_dict(orient="records")

# --------------------------
# Embeddings
# --------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------------
# Chroma
# --------------------------

persist_dir = os.path.join(BASE_DIR, "banking_chromadb")
collection_name = "banking_faqs_poisoned"

client = chromadb.PersistentClient(
    path=persist_dir,
    settings=Settings()
)

existing = [c.name for c in client.list_collections()]

if collection_name in existing:

    print("Loading existing collection...")

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        collection_name=collection_name
    )

else:

    print("Creating collection...")

    vectordb = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=collection_name,
        persist_directory=persist_dir
    )

    vectordb.persist()

# --------------------------
# LangGraph Node
# --------------------------

def retrieve_answer(state: ChatState):

    results = vectordb.similarity_search_with_score(
        state["query"],
        k=1
    )

    if not results:
        return {"answer": "Sorry, I could not find an answer."}

    doc, distance = results[0]

    similarity = 1 - distance

    return {
        "answer":
        f"{doc.metadata['Response']}\n\n"
        f"(Confidence: {similarity:.2f})"
    }

# --------------------------
# Build Graph
# --------------------------

workflow = StateGraph(ChatState)

workflow.add_node("retrieve", retrieve_answer)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", END)

app_graph = workflow.compile()

# --------------------------
# HTML
# --------------------------

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Banking FAQ Chatbot</title>
<style>
body{font-family:Arial;width:900px;margin:auto;margin-top:40px}
textarea{width:100%;height:80px}
.answer{margin-top:20px;padding:15px;border:1px solid #999;background:#f2f2f2}
.examples li{margin:6px 0}
</style>
</head>
<body>

<h2>Banking FAQ Chatbot (Vector Search)</h2>

<form method="post">

<textarea
name="question"
placeholder="Ask a banking question..."></textarea>

<br><br>

<input type="submit" value="Ask">

</form>

<div class="examples">
<h3>Example Questions</h3>
<ul>
<li>How can I open a new bank account?</li>
<li>What documents are required to open an account?</li>
<li>How do I check my balance?</li>
</ul>
</div>

{% if question %}

<h3>Question</h3>
<p>{{question}}</p>

<h3>Answer</h3>

<div class="answer">
{{answer}}
</div>

{% endif %}

</body>
</html>
"""

# --------------------------
# Flask Route
# --------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        result = app_graph.invoke(
            {"query": question}
        )

        answer = result["answer"]

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )

# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
