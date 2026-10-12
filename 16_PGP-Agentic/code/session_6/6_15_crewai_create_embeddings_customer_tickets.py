# Data taken from https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_and_save_embeddings():
    """
    Read customer_tickets.csv, create embeddings from 'body',
    and save Chroma DB with answer + priority metadata
    """
    
    print("Reading customer_tickets.csv...")
    try:
        df = pd.read_csv(os.path.join(BASE_DIR, "..", "..", "..", "4_crewai", "customer_tickets.csv"))
        print(f"Loaded {len(df)} tickets from CSV")
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        print(f"Columns found: {list(df.columns)}")
        
        required_cols = ['body', 'answer', 'priority']
        for col in required_cols:
            if col not in df.columns:
                print("Available columns:", list(df.columns))
                raise ValueError(f"Missing required column: {col}")
        
        # Drop rows with missing or empty 'body'
        df = df.dropna(subset=['body'])
        df = df[df['body'].str.strip() != '']
        print(f"Processing {len(df)} valid tickets...")
        
    except FileNotFoundError:
        print("Error: customer_tickets.csv not found")
        return
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Create documents
    print("Creating documents...")
    documents = []
    for idx, row in df.iterrows():
        body = str(row['body']).strip()
        answer = str(row['answer']).strip() if pd.notna(row['answer']) else ""
        priority = str(row['priority']).strip().lower() if pd.notna(row['priority']) else "unknown"
        
        doc = Document(
            page_content=body,
            metadata={
                "answer": answer,
                "priority": priority,
                "csv_index": idx
            }
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} documents")
    
    # Split documents
    print("Splitting documents...")
    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator=" "
    )
    docs = splitter.split_documents(documents)
    print(f"Split into {len(docs)} document chunks")
    
    # Initialize embeddings
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Clear old database
    persist_dir = os.path.join(BASE_DIR, "customer_support_chroma")
    if os.path.exists(persist_dir):
        print("Clearing previous Chroma database...")
        shutil.rmtree(persist_dir)
    
    # Create Chroma vector DB
    print("Creating embeddings and vector database...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="customer_support"
    )
    
    print(f"Vector database created with {vectordb._collection.count()} vectors!")
    print(f"Saved to: {persist_dir}")
    
def load_embeddings_example():
    """
    Example: load embeddings, run query, fetch best answers + priority
    """
    print("\n" + "="*70)
    print("TESTING: Loading saved embeddings")
    print("="*70)
    
    persist_dir = os.path.join(BASE_DIR, "customer_support_chroma")
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="customer_support"
        )
        
        print(f"Loaded database with {vectordb._collection.count()} vectors\n")
        
        # Test queries
        test_queries = [
            "I forgot my password, how do I reset it?",
            "investment analysis data analytics",
            "security of medical data"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 70)
            results = vectordb.similarity_search(query, k=2)
            
            for i, doc in enumerate(results, 1):
                print(f"\nMatch {i}:")
                print(f"  Body: {doc.page_content[:100]}...")
                print(f"  Answer: {doc.metadata.get('answer', 'N/A')[:100]}...")
                print(f"  Priority: {doc.metadata.get('priority', 'unknown')}")
            
    except Exception as e:
        print(f"Error loading embeddings: {e}")

if __name__ == "__main__":
    create_and_save_embeddings()
    load_embeddings_example()