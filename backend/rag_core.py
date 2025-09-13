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
import numpy as np
import re
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

# --- REFINEMENT 1: A new, balanced prompt encouraging helpful explanations ---
ANSWER_PROMPT = PromptTemplate.from_template(
    """You are a helpful and expert AI assistant. Your task is to provide a clear, well-structured, and explanatory answer to the user's question using ONLY the provided document context.

    **Core Instructions:**
    1.  Analyze the 'Standalone Question' and the 'Document Context' to form your answer.
    2.  Your answer MUST be a helpful explanation that directly addresses the user's question.
    3.  Structure your answer logically. Use paragraphs to separate distinct ideas.
    4.  Be thorough in your explanation, but do not add information that isn't in the context and avoid unnecessary repetition.
    5.  If the context does not contain the information to answer the question, you MUST reply with ONLY this exact phrase: "I could not find an answer to that in the provided documents."

    **Document Context:**
    {context}

    **Standalone Question:** {question}

    **Helpful Answer:**"""
)

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

class ConversationalRAG:
    def __init__(self):
        # --- REFINEMENT 2: Slightly increasing temperature for more natural generation ---
        self.llm = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="not-needed",
            temperature=0.6 # Increased from 0.5 to encourage more explanatory text
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db_directory = "./vector_db"
        self.chroma_client = chromadb.PersistentClient(path=self.db_directory, settings=Settings(anonymized_telemetry=False))
        self.vectorstore = Chroma(client=self.chroma_client, embedding_function=self.embeddings)
        self.rag_chain = None
        self.rag_initialized = False
        self.metadata_file = os.path.join(self.db_directory, "processed_files.json")
        self.processed_files = self._load_processed_files_metadata()

    def _is_follow_up(self, question: str, chat_history: List[Dict]) -> Tuple[bool, List[Tuple[str, str]]]:
        if not chat_history:
            return False, []
        pronoun_pattern = r"\b(it|that|this|those|these|they|them|he|him|she|her)\b"
        if re.search(pronoun_pattern, question.lower()) or len(question.split()) <= 3:
            last_exchange = chat_history[-3:]
            user_msg = last_exchange[0]['content']
            ai_msg = last_exchange[1]['content']
            return True, [(user_msg, ai_msg)]
        question_embedding = self.embeddings.embed_query(question)
        history_embeddings = self.embeddings.embed_documents([turn['content'] for turn in chat_history if turn['role'] == 'user'])
        similarities = [np.dot(question_embedding, hist_emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(hist_emb)) for hist_emb in history_embeddings]
        max_similarity = max(similarities) if similarities else 0
        logger.info(f"Max similarity score with history: {max_similarity:.4f}")
        if max_similarity > 0.7:
            most_relevant_index = np.argmax(similarities)
            user_turn_index = most_relevant_index * 2
            if user_turn_index + 1 < len(chat_history):
                user_msg = chat_history[user_turn_index]['content']
                ai_msg = chat_history[user_turn_index + 1]['content']
                return True, [(user_msg, ai_msg)]
        return False, []

    def initialize_chain(self):
        try:
            if self.chroma_client.get_collection(name="langchain").count() == 0:
                self.rag_initialized = False; return
        except Exception:
            self.rag_initialized = False; return
        logger.info("Initializing custom Conversational RAG chain...")
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
        self.answer_chain = (
            {"context": retriever | _format_docs, "question": RunnablePassthrough()}
            | ANSWER_PROMPT | self.llm | StrOutputParser()
        )
        self.rag_initialized = True
        logger.info("RAG chain components are ready.")

    def query(self, question: str, chat_history: List[Dict]) -> Dict:
        if not self.rag_initialized:
            logger.info("RAG not initialized. Responding with general knowledge.")
            response = self.llm.invoke(question)
            return {"answer": response.content.strip(), "sources": [], "type": "general"}
        is_follow_up, selected_history = self._is_follow_up(question, chat_history)
        final_question = question
        if is_follow_up:
            logger.info("Follow-up detected. Condensing question with selected history.")
            history_str = "\n".join([f"Human: {h}\nAI: {a}" for h, a in selected_history])
            final_question = (
                CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
            ).invoke({"chat_history": history_str, "question": question})
            logger.info(f"Standalone question: {final_question}")
        else:
            logger.info("No follow-up detected. Using original question.")
        response = self.answer_chain.invoke(final_question)
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