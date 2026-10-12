import os
from openai import OpenAI
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# 1. Load environment & OpenAI client
# --------------------------------------------------
load_dotenv(override=True)
client = OpenAI()

# --------------------------------------------------
# 2. Read FAQ file
# --------------------------------------------------
FAQ_PATH = os.path.join(BASE_DIR, "faq.txt")
faq_text = Path(FAQ_PATH).read_text(encoding="utf-8")

# --------------------------------------------------
# 3. OpenAI embedding function for Chroma
# --------------------------------------------------
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=None,              # uses OPENAI_API_KEY from .env
    model_name="text-embedding-3-small"
)

# --------------------------------------------------
# 4. Create / Load ChromaDB (LOCAL storage)
# --------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=os.path.join(BASE_DIR, "chroma_faq_db")
)

collection = chroma_client.get_or_create_collection(
    name="faqs",
    embedding_function=openai_ef
)

# --------------------------------------------------
# 5. Add FAQ document to Chroma
# --------------------------------------------------
collection.add(
    documents=[faq_text],
    ids=["faq-1"]
)

#chroma_client.persist()
print("FAQ stored in local ChromaDB")

# --------------------------------------------------
# 6. Query ChromaDB
# --------------------------------------------------
user_question = "What if I get a damaged product?"

results = collection.query(
    query_texts=[user_question],
    n_results=1
)

retrieved_context = results["documents"][0][0]

# --------------------------------------------------
# 7. Send retrieved context to OpenAI for answer
# --------------------------------------------------
response = client.responses.create(
    model="gpt-4o-mini",
    input=f"""
Answer the question using the context below.

Context:
{retrieved_context}

Question:
{user_question}
"""
)

print("\nBot Answer:")
print(response.output_text)
