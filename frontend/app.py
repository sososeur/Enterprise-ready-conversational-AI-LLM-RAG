# file: frontend/app.py

import streamlit as st
import requests
import json
import os

API_BASE_URL = "http://localhost:8000"
CHAT_HISTORY_DIR = "chat_history"

def save_chat_session(session_name, messages):
    if not os.path.exists(CHAT_HISTORY_DIR): os.makedirs(CHAT_HISTORY_DIR)
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_name}.json")
    with open(file_path, "w") as f: json.dump(messages, f, indent=2)

def load_chat_session(session_name):
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{session_name}.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f: return json.load(f)
    return []

def get_saved_sessions():
    if not os.path.exists(CHAT_HISTORY_DIR): return []
    return sorted([f.replace(".json", "") for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")])

def check_rag_status():
    try:
        response = requests.get(f"{API_BASE_URL}/status")
        st.session_state.rag_initialized = response.status_code == 200 and response.json().get("rag_initialized", False)
    except requests.ConnectionError:
        st.session_state.rag_initialized = False

st.set_page_config(page_title="Conversational AI", layout="wide")
st.title("🤖 Conversational AI Assistant")

if "messages" not in st.session_state: st.session_state.messages = []
if "current_chat" not in st.session_state: st.session_state.current_chat = "main_chat"
if "rag_initialized" not in st.session_state: check_rag_status()

with st.sidebar:
    st.header("⚙️ System Management")
    if st.session_state.rag_initialized:
        st.success("✅ RAG System is Ready")
    else:
        st.warning("⚠️ RAG System Not Ready")
    
    if st.button("🔄 Process Knowledge Base"):
        with st.spinner("Processing documents..."):
            try:
                response = requests.post(f"{API_BASE_URL}/process-documents")
                if response.status_code == 200:
                    resp_data = response.json()
                    st.success(resp_data.get("message", "Success!"))
                    check_rag_status()
                else:
                    st.error(f"Error: {response.text}")
            except requests.ConnectionError:
                st.error("Connection Error: Could not connect to the backend.")
    
    st.markdown("---")
    st.header("💬 Chat Sessions")

    # --- REFINEMENT: Custom Chat Naming ---
    with st.form("new_chat_form", clear_on_submit=True):
        new_chat_name = st.text_input("New chat name:")
        submitted = st.form_submit_button("Create Chat")
        if submitted and new_chat_name:
            if new_chat_name in get_saved_sessions():
                st.warning("Chat name already exists!")
            else:
                st.session_state.current_chat = new_chat_name
                st.session_state.messages = []
                save_chat_session(new_chat_name, []) # Save empty chat to list it
                st.rerun()
    # --- END OF REFINEMENT ---

    st.markdown("##### Saved Chats")
    for session in get_saved_sessions():
        if st.button(session, key=f"session_{session}", use_container_width=True):
            st.session_state.current_chat = session
            st.session_state.messages = load_chat_session(session)
            st.rerun()

st.header(f"Conversation: `{st.session_state.current_chat}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.info(f"Source: {source.get('source', 'N/A')}\n\nContent: {source.get('content', 'N/A')}")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history_for_api = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages[:-1]]
    api_payload = {"question": prompt, "chat_history": history_for_api}
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(f"{API_BASE_URL}/chat", json=api_payload)
                response.raise_for_status()
                assistant_response = response.json()
                answer = assistant_response.get("answer", "An error occurred.")
                sources = assistant_response.get("sources", [])
                
                st.markdown(answer)
                if sources:
                     with st.expander("View Sources"):
                        for source in sources:
                            st.info(f"Source: {source.get('source', 'N/A')}\n\nContent: {source.get('content', 'N/A')}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                
            except requests.exceptions.RequestException as e:
                error_detail = "Could not connect to the backend."
                if e.response is not None:
                    try: error_detail = e.response.json().get('detail', e.response.text)
                    except json.JSONDecodeError: error_detail = e.response.text
                st.error(f"API Error: {error_detail}")
                st.session_state.messages.append({"role": "assistant", "content": "Error: Could not get a response."})

    save_chat_session(st.session_state.current_chat, st.session_state.messages)