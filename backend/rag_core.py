# file: backend/rag_core.py

import os
import json
import hashlib
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:"""
)

ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a highly skilled AI assistant designed for concise and accurate answers.

    **CRITICAL INSTRUCTIONS:**
    1.  Your primary goal is to answer the user's 'Standalone Question' directly and concisely.
    2.  Use the provided 'Document Context' as your ONLY source of truth.
    3.  **DO NOT** repeat information. Synthesize the key points into a brief, focused answer. Avoid verbose explanations.
    4.  If the context does not contain the answer, you MUST reply with ONLY this exact phrase: "I could not find an answer to that in the provided documents."
    5.  Focus exclusively on the information needed to answer the 'Standalone Question'. Ignore any irrelevant details in the context.

    **Document Context:**
    {context}

    **Standalone Question:** {question}

    **Concise Answer:**"""
)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = []
    for human, ai in chat_history:
        buffer.append(f"Human: {human}\nAI: {ai}")
    return "\n".join(buffer)

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

class ConversationalRAG:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            temperature=0.5
            
        
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db_directory = "./vector_db"
        self.chroma_client = chromadb.PersistentClient(path=self.db_directory, settings=Settings(anonymized_telemetry=False))
        self.vectorstore = Chroma(client=self.chroma_client, embedding_function=self.embeddings)
        self.rag_chain = None
        self.rag_initialized = False
        self.metadata_file = os.path.join(self.db_directory, "processed_files.json")
        self.processed_files = self._load_processed_files_metadata()

    def initialize_chain(self):
        try:
            if self.chroma_client.get_collection(name="langchain").count() == 0:
                self.rag_initialized = False; return
        except Exception:
            self.rag_initialized = False; return
            
        logger.info("Initializing stateless Conversational RAG chain...")
        
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

        # --- REFINEMENT 1: Removed the stateful ConversationBufferWindowMemory ---
        # The chain is now fully stateless and relies on the history passed in each call.

        inputs = RunnablePassthrough()
        
        condense_question_chain = (
            {"question": lambda x: x["question"], "chat_history": lambda x: _format_chat_history(x["chat_history"])}
            | CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
        )

        answer_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough(), "chat_history": RunnableLambda(lambda x: "")}
            | ANSWER_PROMPT | self.llm | StrOutputParser()
        )

        self.rag_chain = condense_question_chain | answer_chain
        
        self.rag_initialized = True
        logger.info("Stateless RAG chain is ready.")

    def query(self, question: str, chat_history: List[Dict]) -> Dict:
        if not self.rag_initialized or not self.rag_chain:
            logger.info("RAG not initialized. Responding with general knowledge.")
            response = self.llm.invoke(question)
            return {"answer": response.content.strip(), "sources": [], "type": "general"}

        logger.info("RAG is initialized. Routing to custom RAG chain.")
        
        # --- REFINEMENT 2: Simplified and more robust history formatting ---
        # Assumes history is always [user, assistant, user, assistant, ...]
        formatted_history_tuples = []
        for i in range(0, len(chat_history), 2):
            if i + 1 < len(chat_history):
                user_msg = chat_history[i]['content']
                ai_msg = chat_history[i+1]['content']
                formatted_history_tuples.append((user_msg, ai_msg))
        # --- End of Refinement ---

        response = self.rag_chain.invoke({
            "question": question,
            "chat_history": formatted_history_tuples
        })
        
        return {"answer": response.strip(), "sources": [], "type": "rag"}

    # --- No changes to the functions below this line ---

    def load_and_process_documents(self, directory="knowledge_base") -> int:
        logger.info(f"Scanning for documents in directory: '{os.path.abspath(directory)}'")
        if not os.path.exists(directory): return 0
        all_files = [os.path.join(root, file) for root, _, files in os.walk(directory) for file in files if file.endswith(('.pdf', '.txt', '.md'))]
        files_to_process = [fp for fp in all_files if self.processed_files.get(os.path.abspath(fp), {}).get('hash') != self._get_file_hash(fp)]
        if not files_to_process: return 0
        logger.info(f"Processing {len(files_to_process)} new/changed files...")
        documents = []
        for file_path in files_to_process:
            try:
                if file_path.endswith('.pdf'): loader = PyPDFLoader(file_path)
                elif file_path.endswith('.txt'): loader = TextLoader(file_path)
                else: loader = UnstructuredMarkdownLoader(file_path)
                documents.extend(loader.load())
                self.processed_files[os.path.abspath(file_path)] = {'hash': self._get_file_hash(file_path)}
            except Exception as e:
                logger.error(f"Failed to load '{os.path.basename(file_path)}': {e}. Skipping.")
        if not documents: return 0
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        if chunks:
            self.update_vector_store(chunks)
            self._save_processed_files_metadata()
            return len(chunks)
        return 0

    def update_vector_store(self, chunks: List):
        logger.info(f"Adding {len(chunks)} new chunks to vector store...")
        self.vectorstore.add_documents(chunks)
        logger.info("Vector store updated successfully.")

    def _load_processed_files_metadata(self):
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f: return json.load(f)
        except Exception: pass
        return {}

    def _save_processed_files_metadata(self):
        try:
            os.makedirs(self.db_directory, exist_ok=True)
            with open(self.metadata_file, 'w') as f: json.dump(self.processed_files, f, indent=2)
        except Exception as e: logger.error(f"Error saving metadata: {e}")

    def _get_file_hash(self, file_path: str) -> str:
        with open(file_path, 'rb') as f: return hashlib.md5(f.read()).hexdigest()

