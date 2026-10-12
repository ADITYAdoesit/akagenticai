import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(override=True)

persist_directory = os.path.join(BASE_DIR, "chroma_db")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if os.path.exists(persist_directory) and os.listdir(persist_directory):
    print("Chroma vector store already exists, loading it instead of re-embedding")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
else:
    # Load the PDF from URL
    pdf_url = "https://www.adobe.com/be_en/active-use/pdf/Alice_in_Wonderland.pdf"
    loader = PyPDFLoader(pdf_url)
    pages = loader.load()

    print(f"Loaded {len(pages)} pages from the PDF")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings and vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    print(f"Created Chroma vector store with {len(chunks)} documents")

# Create the RAG chain using LCEL (LangChain Expression Language)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask questions about Alice in Wonderland
questions = [
    "Who is the main character and what happens at the beginning of the story?",
    "What did the Caterpillar ask Alice?",
    "Describe the Mad Hatter's tea party.",
    "What happened at the trial at the end of the story?",
]

for question in questions:
    answer = rag_chain.invoke(question)
    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 60)