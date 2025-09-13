## Enterprise-Ready Conversational AI: A RAG-Powered Chatbot for Internal Knowledge Management

Developed a secure, on-premise conversational AI assistant to enable employees to query internal documentation through a natural language interface. This project leverages a custom Retrieval-Augmented Generation (RAG) pipeline used to load relevent documents from the vector database chromadb based on query and context, orchestrated with LangChain, to ground a locally-hosted Llama 3 8B Instruct model in a private knowledge base. The system features a decoupled three-tier architecture with a FastAPI backend for AI logic and a dynamic Streamlit frontend for user interaction, ensuring scalability and maintainability.

![Architecture Diagram](https://i.imgur.com/your-architecture-diagram.png)  <!-- You can create a simple diagram and upload it to a site like Imgur to get a link -->

## Table of Contents

- [Key Features](#key-features)
- [Core Technologies](#core-technologies)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Use](#how-to-use)
- [Future Enhancements](#future-enhancements)

## Key Features

-   **Conversational Memory:** Remembers the full context of the current chat session, allowing for natural follow-up questions.
-   **On-Demand Knowledge Ingestion:** Users can trigger the processing of new or updated documents directly from the web interface.
-   **Intelligent Query Handling:** Utilizes the LLM to rewrite user queries for better semantic search and to condense conversation history for accurate context.
-   **Strictly Factual Responses:** The RAG pipeline is engineered with a strict prompt that forces the LLM to answer *only* from the provided documents, preventing hallucinations.
-   **Modular & Scalable:** The decoupled frontend/backend architecture allows for independent development, scaling, and maintenance.
-   **100% On-Premise & Secure:** All components, including the LLM, run locally. No company data ever leaves the local machine.

## Core Technologies

| Component             | Technology                                                                                                | Role                                                                                                                              |
| --------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Language Model**    | **Llama 3 8B Instruct (via LM Studio)**                                                                   | The core reasoning engine for understanding text, rewriting queries, and synthesizing answers.                                    |
| **AI Orchestration**  | **LangChain**                                                                                             | The toolkit used to build and orchestrate the entire RAG pipeline, connecting prompts, models, and the vector database.           |
| **Vector Database**   | **ChromaDB**                                                                                              | A specialized database that stores document embeddings (vectors) and performs efficient semantic similarity searches.             |
| **Embedding Model**   | **`sentence-transformers/all-MiniLM-L6-v2`**                                                              | A high-performance model that converts text chunks into meaningful numerical vectors (embeddings).                                |
| **Backend Framework** | **FastAPI**                                                                                               | Provides a high-performance, robust API server to handle AI logic and serve requests from the frontend.                           |
| **Frontend Framework**| **Streamlit**                                                                                             | Enables the rapid development of the interactive, real-time chat interface.                                                       |

## System Architecture & Workflow

The application operates on a sophisticated, multi-step RAG pipeline that ensures accurate, contextual, and relevant answers.

**1. Document Ingestion (On-Demand):**
   - An administrator places PDF, TXT, or MD files into the `knowledge_base/` directory.
   - Through the Streamlit UI, they trigger the `/process-documents` API endpoint.
   - The backend scans the directory, identifies new or changed files via MD5 hashing, and loads them.
   - Documents are split into smaller, overlapping chunks.
   - The `HuggingFaceEmbeddings` model converts each chunk into a vector embedding.
   - These embeddings are stored in the local **ChromaDB** vector store for fast retrieval.

**2. Conversational Query Workflow:**
   - **User Input:** An employee asks a question in the Streamlit chat interface.
   - **History Condensation:** The FastAPI backend receives the question and the *entire* chat history. The first LLM chain (**`condense_question_chain`**) synthesizes this information into a clear, standalone question. This is critical for understanding follow-ups.
   - **Document Retrieval:** The standalone question is then used to query the ChromaDB. The retriever uses a **Maximal Marginal Relevance (MMR)** algorithm to find a set of documents that are both relevant to the question and diverse in content.
   - **Answer Generation:** The retrieved document chunks and the standalone question are passed to the second LLM chain (**`answer_chain`**). A highly restrictive prompt commands the LLM to synthesize a concise, non-repetitive answer *based only on the provided context*.
   - **Response to User:** The final answer is sent back to the Streamlit frontend and displayed to the user.

## Project Structure

The project is organized into a clean, three-tier structure:

```
LLM_development/
├── backend/
│   ├── __init__.py
│   └── rag_core.py         # Core RAG pipeline, prompts, and AI logic
│   └── api.py              # FastAPI server, endpoints, and request handling
│
├── frontend/
│   └── app.py              # Streamlit application UI and logic
│
├── knowledge_base/         # <-- Place your PDF, TXT, and MD files here
│
├── vector_db/              # (Auto-generated) ChromaDB persistent storage
│
├── chat_history/           # (Auto-generated) Saved JSON chat logs
│
├── requirements.txt        # All Python dependencies
└── run.sh                  # Script to start the backend and frontend servers
```

## Setup & Installation

Follow these steps to get the application running locally.

**Prerequisites:**
-   Python 3.9+
-   [LM Studio](https://lmstudio.ai/) installed and running.

**Step 1: Download the LLM in LM Studio**
1.  Open LM Studio.
2.  Search for and download a **Llama 3 8B Instruct GGUF** model (e.g., from QuantFactory).
3.  Go to the "Local Server" tab (the `<- ->` icon).
4.  Select your downloaded model at the top and click **"Start Server"**.

**Step 2: Clone & Set Up the Project**
```bash
# Clone the repository
git clone <your-repo-url>
cd LLM_development

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all required dependencies
pip install -r requirements.txt
```

**Step 3: Add Your Knowledge Base**
1.  Create the `knowledge_base` directory if it doesn't exist.
2.  Add your PDF, TXT, and/or Markdown files to this directory.

**Step 4: Make the Run Script Executable**
This only needs to be done once.
```bash
chmod +x run.sh
```

## How to Use

**1. Start the Application:**
   - Make sure your LM Studio server is running.
   - In your terminal, from the `LLM_development` root directory, run the start script:
   ```bash
   ./run.sh
   ```
   - This will start the FastAPI backend and then launch the Streamlit app in your web browser.

**2. Process Your Documents:**
   - In the Streamlit web interface, you will see a "System Management" section in the sidebar.
   - The status will show "RAG System Not Ready".
   - Click the **"Process Knowledge Base"** button. The system will ingest your documents and build the vector database.
   - When complete, the status will change to "RAG System is Ready".

**3. Start Chatting:**
   - Create a new chat session using the form in the sidebar.
   - Ask questions about the content of your documents.
   - Ask follow-up questions to test the conversational memory.

## Future Enhancements

-   **Source Highlighting:** Modify the chain to return the specific source documents and display them in the UI for verification.
-   **User Authentication:** Implement a login system to manage access and separate chat histories by user.
-   **Streaming Responses:** Use FastAPI WebSockets and Streamlit's `st.write_stream` to stream the LLM's response token-by-token for a more interactive feel.
-   **Deployment:** Containerize the application using Docker for easier deployment on a dedicated internal server.
