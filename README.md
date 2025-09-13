## Enterprise-Ready Conversational AI: A RAG-Powered Chatbot for Internal Knowledge Management

This repository contains the prototype for a secure, on-premise conversational AI assistant, designed to address the operational challenge of internal information retrieval. The system enables employees to query enterprise documentation via a natural language interface.

This project implements a custom, multi-step **Retrieval-Augmented Generation (RAG)** pipeline, orchestrated with LangChain, to ground a locally-hosted Llama 3.2B Instruct model in a private knowledge base. The system is built on a decoupled three-tier architecture, proving the viability of leveraging local LLMs to transform static documents into an interactive, on-demand knowledge resource and enhance operational efficiency.

## Table of Contents

- [Key Features of the Prototype](#key-features-of-the-prototype)
- [Core Technologies](#core-technologies)
- [System Architecture & Workflow](#system-architecture--workflow)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Use](#how-to-use)
- [Future Enhancements & Scalability](#future-enhancements--scalability)

## Key Features of the Prototype

This prototype successfully implements several advanced features crucial for a truly useful conversational AI:

-   **Stateful Conversational Memory:** Maintains the full context of a chat session by utilizing a history-condensation chain, enabling coherent, multi-turn dialogues.
-   **Dynamic Knowledge Base Ingestion:** The knowledge base can be updated in real-time. The ingestion and vectorization pipeline is triggered via an API endpoint from the UI.
-   **Adaptive Query Processing:** Employs a sophisticated two-stage process involving history-aware query condensation and semantic retrieval to accurately interpret user intent.
-   **Controlled and Grounded Generation:** A highly-tuned prompt architecture combined with model parameters (`temperature`, `repeat_penalty`) ensures responses are explanatory yet concise, and strictly grounded in the retrieved document context to mitigate hallucination.
-   **Modular Three-Tier Architecture:** A decoupled Streamlit frontend, FastAPI backend, and local data layer ensure the system is maintainable and ready for future scaling.
-   **Fully On-Premise and Secure:** All components, including the LLM, operate locally. No proprietary data ever leaves the local machine, guaranteeing complete privacy and security.

## Core Technologies

| Component             | Technology                                   | Role                                                                    |
| --------------------- | -------------------------------------------- | ----------------------------------------------------------------------- |
| **Language Model**    | **Llama 3.2B Instruct (via LM Studio)**      | The core reasoning engine for understanding and generating language.      |
| **AI Orchestration**  | **LangChain**                                | The framework for orchestrating the RAG pipeline and its components.    |
| **Vector Database**   | **ChromaDB**                                 | Stores document embeddings for efficient semantic search.                 |
| **Embedding Model**   | **`all-MiniLM-L6-v2`**                       | Converts text into meaningful numerical vectors (embeddings).           |
| **Backend Framework** | **FastAPI**                                  | Provides a high-performance API server for all AI logic.                |
| **Frontend Framework**| **Streamlit**                                | Enables the rapid development of the interactive chat interface.        |

## System Architecture & Workflow

This prototype's intelligence lies in its custom-built, adaptive RAG workflow.

#### 1. Document Ingestion (On-Demand)

-   Documents (PDF, TXT, MD) are staged in the `knowledge_base/` directory.
-   An API call to the `/process-documents` endpoint initiates the ingestion pipeline.
-   The pipeline identifies novel or modified files via MD5 checksum comparison.
-   Documents are segmented into overlapping text chunks.
-   Each chunk is vectorized by the embedding model and indexed in the ChromaDB vector store.

#### 2. Conversational Query Workflow

The system follows a sophisticated, multi-step process for each user query:

1.  **Step 1: Follow-up Triage & Context Selection:** The system algorithmically determines if the incoming query is a follow-up. It combines linguistic heuristic checks (e.g., pronoun detection) with semantic similarity scores between the new query embedding and the historical turn embeddings.

2.  **Step 2: Standalone Question Generation:** If a follow-up is detected, the most relevant historical exchange is selected. This context, along with the new query, is passed to the `CONDENSE_QUESTION_PROMPT` chain to generate a self-contained, standalone question that captures the full user intent.

3.  **Step 3: Context Retrieval:** The standalone question is used to query ChromaDB. The retriever employs a **Maximal Marginal Relevance (MMR)** algorithm to fetch a contextually relevant and diverse set of document chunks.

4.  **Step 4: Grounded Response Synthesis:** The retrieved context and the standalone question are passed to the final LLM chain, which utilizes the `ANSWER_PROMPT`. This prompt instructs the model to synthesize a helpful, non-repetitive, and explanatory answer that is strictly grounded in the provided context.

## Project Structure

```
LLM_development/
├── backend/
│   ├── __init__.py
│   └── rag_core.py         # Core RAG pipeline, prompts, and AI logic
│   └── api.py              # FastAPI server and endpoints
│
├── frontend/
│   └── app.py              # Streamlit application UI
│
├── knowledge_base/         # <-- Place your documents here
│
├── vector_db/              # (Auto-generated) ChromaDB storage
│
├── chat_history/           # Contains chat histories of the model's responses.
│
├── requirements.txt        # Python dependencies
└── run.sh                  # Script to start the application
```

## Setup & Installation

**Prerequisites:**
-   Python 3.9+
-   [LM Studio](https://lmstudio.ai/) installed.

**Step 1: Download the LLM in LM Studio**
1.  Open LM Studio, search for, and download a **Llama 3.2B Instruct GGUF** model.
2.  Navigate to the "Local Server" tab (`<- ->` icon).
3.  Select your downloaded model and click **"Start Server"**.

**Step 2: Set Up the Project**
```bash
# Clone the repository
git clone <your-repo-url>
cd LLM_development

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt```

**Step 3: Add Your Knowledge Base**
1.  Add your PDF, TXT, and/or Markdown files to the `knowledge_base/` directory.

**Step 4: Make the Run Script Executable**
```bash
chmod +x run.sh
```

## How to Use

1.  **Start the Application:** Ensure your LM Studio server is running. Then, from the project's root directory, run:
    ```bash
    ./run.sh
    ```
2.  **Process Documents:** In the Streamlit UI that opens, click the **"Process Knowledge Base"** button in the sidebar. Wait for the status to change to "RAG System is Ready".
3.  **Chat:** Create a custom-named chat session and begin asking questions about your documents.

## Future Enhancements & Scalability

This prototype successfully establishes a strong foundation. The next steps to evolve this into a production-grade enterprise solution include:

-   **Production-level Deployment:** Deployment of the conversational chatbot on the enterprise server with all the necessary configurations like processing, RAM, storage where it can be facilitated to access for employees.
-    **Enterprise Authentication (OAuth/SSO):** Employ authentication systems to manage user-specific chat histories and control access.
-    **Containerization:** Package the backend, frontend, and data stores using **Docker** for scalable and reproducible deployment on internal infrastructure.
-   **Real-Time Response Streaming:** Implement WebSockets to stream the LLM's response token-by-token for a more interactive and responsive user experience.
-   **Source Citation and Verification:** Modify the RAG chain to return the specific source documents used for an answer and display them in the UI for user verification and trust.

