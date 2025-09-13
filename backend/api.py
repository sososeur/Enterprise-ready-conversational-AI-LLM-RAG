# file: backend/api.py

from asyncio.log import logger
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager
from backend.rag_core import ConversationalRAG

rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    print("Initializing RAG System on startup...")
    rag_system = ConversationalRAG()
    rag_system.initialize_chain() # Will gracefully handle if DB doesn't exist
    yield
    print("Shutting down API.")

app = FastAPI(
    title="Conversational RAG API",
    description="An API for interacting with the RAG chatbot.",
    version="1.0.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    question: str
    # This is more flexible and accepts any dictionary structure in the list
    chat_history: List[Dict] 

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    type: str # 'rag', 'general', or 'error'

# --- New Endpoint: Get RAG Status ---
@app.get("/status")
async def get_status():
    if rag_system:
        return {"rag_initialized": rag_system.rag_initialized}
    return {"rag_initialized": False}

# --- New Endpoint: Trigger Document Processing ---
@app.post("/process-documents")
async def process_documents():
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not available.")
    try:
        logger.info("API call received to process documents.")
        chunks_added = rag_system.load_and_process_documents()
        
        # Re-initialize the chain to load the new data
        if chunks_added > 0:
            rag_system.initialize_chain()
            
        message = f"Processing complete. Added {chunks_added} new document chunks." if chunks_added > 0 else "Processing complete. No new documents found."
        return {"message": message, "chunks_added": chunks_added}
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    response = rag_system.query(request.question, request.chat_history)
    return response