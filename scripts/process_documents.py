# file: scripts/process_documents.py

import sys
import os

# Add the parent directory to the Python path to allow imports from `backend`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.rag_core import ConversationalRAG

def main():
    print("Starting document processing...")
    
    # Define the knowledge base directory relative to this script's location
    knowledge_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo_knowledge_base'))
    
    if not os.path.exists(knowledge_base_dir):
        print(f"Error: Knowledge base directory not found at '{knowledge_base_dir}'")
        print("Please create it and add your documents.")
        return

    # Initialize the RAG system
    rag_system = ConversationalRAG()
    
    # 1. Load and process new/changed documents
    new_chunks = rag_system.load_and_process_documents(directory=knowledge_base_dir)
    
    # 2. Update the vector store with the new chunks
    if new_chunks:
        rag_system.update_vector_store(new_chunks)
        print("Document processing complete. Vector store has been updated.")
    else:
        print("No new documents to process. Vector store is up-to-date.")

if __name__ == "__main__":
    main()