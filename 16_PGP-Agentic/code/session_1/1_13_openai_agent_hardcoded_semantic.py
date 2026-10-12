import os
# pip install openai python-dotenv chromadb openai-agents sentence-transformers

from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, function_tool
import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

client = OpenAI()

# ----------------------------------------------------
# 1. Initialize ChromaDB
# ----------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "rag", "chromadb")
)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection_name = "faq_support"

# Delete collection during development
try:
    chroma_client.delete_collection(collection_name)
except:
    pass

collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_fn
)

# ----------------------------------------------------
# 2. Knowledge Base
# ----------------------------------------------------

faq_data = [
    {
        "id": "1",
        "question": "How long does shipping take?",
        "answer": "Our standard shipping time is 3-5 business days."
    },
    {
        "id": "2",
        "question": "What is your return policy?",
        "answer": "You can return any product within 30 days of delivery."
    },
    {
        "id": "3",
        "question": "Do products have a warranty?",
        "answer": "All products come with a one-year warranty covering manufacturing defects."
    },
    {
        "id": "4",
        "question": "Which payment methods do you accept?",
        "answer": "We accept credit cards, debit cards, and PayPal."
    },
    {
        "id": "5",
        "question": "How can I contact customer support?",
        "answer": "You can reach our support team 24/7 via email or chat."
    }
]

collection.add(
    ids=[item["id"] for item in faq_data],
    documents=[item["question"] for item in faq_data],
    metadatas=[
        {
            "answer": item["answer"]
        }
        for item in faq_data
    ]
)

print("Knowledge base loaded.")

# ----------------------------------------------------
# 3. RAG Tool
# ----------------------------------------------------

@function_tool
async def faq_invoker(question: str) -> str:
    """
    Search FAQ knowledge base using semantic search.
    """

    result = collection.query(
        query_texts=[question],
        n_results=1
    )

    if len(result["ids"][0]) == 0:
        return "Sorry, I couldn't find an answer."

    answer = result["metadatas"][0][0]["answer"]

    return answer

# ----------------------------------------------------
# 4. Agent
# ----------------------------------------------------

faq_agent = Agent(
    name="Customer Support Bot",
    instructions="""
You are a helpful customer support assistant.

Whenever the user asks anything related to
shipping, returns, warranty, payment or support,
ALWAYS use the FAQ search tool.

Reply naturally using the tool output.
""",
    tools=[faq_invoker]
)

# ----------------------------------------------------
# 5. Chat Function
# ----------------------------------------------------

async def chat(message):
    result = await Runner.run(faq_agent, message)
    return result.final_output

# ----------------------------------------------------
# 6. Interactive Chat
# ----------------------------------------------------

async def main():

    print("=" * 50)
    print("Customer Support Bot")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:

        user = input("\nYou: ")

        if user.lower() == "exit":
            break

        response = await chat(user)

        print("\nBot:", response)


if __name__ == "__main__":
    asyncio.run(main())