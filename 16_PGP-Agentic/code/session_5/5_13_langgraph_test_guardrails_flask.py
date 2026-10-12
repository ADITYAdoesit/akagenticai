# pip install flask langchain-community langchain-huggingface langchain-anthropic python-dotenv

from flask import Flask, request, render_template_string
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain_anthropic import ChatAnthropic
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv()

app = Flask(__name__)

# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------
# Guardrail Vector DB
# ------------------------------------------------------------
GUARDRAIL_DB_PATH = os.path.join(BASE_DIR, "guardrail_chromadb")

guardrail_db = Chroma(
    persist_directory=GUARDRAIL_DB_PATH,
    embedding_function=embeddings
)

# ------------------------------------------------------------
# Retrieve top-k closest matches
# ------------------------------------------------------------
def get_top_matches(query: str, k: int = 3):
    return guardrail_db.similarity_search_with_score(query, k=k)

# ------------------------------------------------------------
# Chat logic
# ------------------------------------------------------------
def chat_fn(message):
    results = get_top_matches(message, k=3)

    if not results:
        return "No similar guardrail examples found."

    output = [
        "### Closest Guardrail Matches\n"
    ]

    for i, (doc, distance) in enumerate(results, start=1):
        category = doc.metadata.get("category", "unknown")
        text_preview = doc.page_content[:300].replace("\n", " ")

        output.append(
            f"**Match {i}**\n"
            f"- **Category:** `{category}`\n"
            f"- **Cosine distance:** `{distance:.4f}`\n"
            f"- **Example text:** {text_preview}...\n"
        )

    return "\n".join(output)

# ------------------------------------------------------------
# HTML
# ------------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Guardrail Vector DB Inspector</title>
<style>
body{font-family:Arial;width:900px;margin:auto;margin-top:40px}
textarea{width:100%;height:80px}
.answer{margin-top:20px;padding:15px;border:1px solid #999;background:#f2f2f2;white-space:pre-wrap}
.examples li{margin:6px 0}
</style>
</head>
<body>

<h2>Guardrail Vector DB Inspector</h2>
<p>Shows top-3 semantic matches from the guardrail dataset</p>

<form method="post">

<textarea
name="question"
placeholder="Type a query to test against the guardrail dataset..."></textarea>

<br><br>

<input type="submit" value="Check">

</form>

<div class="examples">
<h3>Example Queries</h3>
<ul>
<li>how to steal a password</li>
<li>how to use a gun</li>
<li>explain prompt injection</li>
<li>write ransomware code</li>
<li>how to stay motivated</li>
</ul>
</div>

{% if question %}

<h3>Query</h3>
<p>{{question}}</p>

<h3>Result</h3>

<div class="answer">
{{answer}}
</div>

{% endif %}

</body>
</html>
"""

# ------------------------------------------------------------
# Flask Route
# ------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    question = ""
    answer = ""

    if request.method == "POST":

        question = request.form["question"]

        answer = chat_fn(question)

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
