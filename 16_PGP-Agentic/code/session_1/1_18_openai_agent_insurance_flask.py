# pip install flask pypdf python-dotenv openai chromadb openai-agents

import os
import asyncio
from flask import Flask, request, render_template_string
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI
from agents import Agent, Runner, function_tool
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = os.path.join(BASE_DIR, "Introduction_to_Insurance.pdf")

reader = PdfReader(PDF_PATH)
pdf_text = "".join(page.extract_text() or "" for page in reader.pages)

def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

pdf_chunks = split_text(pdf_text)

chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "rag", "chroma_pdf")
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection_name="pdf_collection"

existing=[c.name for c in chroma_client.list_collections()]

if collection_name in existing:
    collection=chroma_client.get_collection(collection_name,
        embedding_function=embedding_fn)
else:
    collection=chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    collection.add(
        documents=pdf_chunks,
        ids=[f"chunk_{i}" for i in range(len(pdf_chunks))]
    )

@function_tool
async def get_pdf_answer(query:str)->str:
    result=collection.query(query_texts=[query],n_results=1)

    if result["documents"] and result["documents"][0]:
        chunk=result["documents"][0][0]

        prompt=f"""
You are a helpful assistant answering insurance questions.

Use ONLY the following document.

Document:
{chunk}

Question:
{query}

Answer:
"""

        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.1
        )

        return response.choices[0].message.content.strip()

    return "No relevant information found."

rag_agent=Agent(
    name="PDF RAG Bot",
    instructions="""
Answer questions using the PDF tool only.
Do not invent information.
""",
    tools=[get_pdf_answer]
)

async def ask(question):
    result=await Runner.run(rag_agent,question)
    return result.final_output

HTML="""
<!doctype html>
<html>
<head>
<title>Insurance PDF RAG Bot</title>
<style>
body{font-family:Arial;width:900px;margin:auto;margin-top:30px}
textarea{width:100%;height:80px}
.answer{margin-top:20px;padding:15px;background:#f2f2f2;border:1px solid #999}
</style>
</head>
<body>
<h2>Insurance PDF RAG Bot</h2>

<form method="post">
<textarea name="question" placeholder="Ask a question about insurance..."></textarea><br><br>
<input type="submit" value="Ask">
</form>

{% if question %}
<h3>Question</h3>
<p>{{question}}</p>

<h3>Answer</h3>
<div class="answer">{{answer}}</div>
{% endif %}

</body>
</html>
"""

@app.route("/",methods=["GET","POST"])
def home():

    question=""
    answer=""

    if request.method=="POST":
        question=request.form["question"]
        answer=asyncio.run(ask(question))

    return render_template_string(
        HTML,
        question=question,
        answer=answer
    )

if __name__=="__main__":
    app.run(debug=True)
