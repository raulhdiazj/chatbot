### Backend file

import os
import json
import time
from datetime import datetime
from typing import List, Dict

# LangChain & Hugging Face Imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document

# --- Configuration & Paths ---
DB_DIR = "./chroma_db"
LOG_FILE = "colabbuddy_analytics.jsonl"
PROFILE_FILE = "user_profiles.json"

# --- 1. Knowledge Base Initialization ---
def initialize_knowledge_base():
    """
    Ingests mock documentation, chunks it, embeds it with BGE-Small, 
    and persists it locally to Chroma DB.
    """
    os.environ["ANONYMIZED_TELEMETRY"] = "False" # Suppress telemetry warnings
    
    kb_docs = [
        Document(
            page_content="Google Colab Free Tier allocates a T4 GPU subject to availability. The maximum continuous runtime lifetime of a single notebook session is 12.5 hours. Sessions can timeout much earlier due to user inactivity or idle kernel triggers.",
            metadata={"source": "colab_limits_faq.md", "topic": "runtime_limits"}
        ),
        Document(
            page_content="When encountering 'CUDA out of memory' (OOM) or system RAM crashes in Colab, mitigate by: 1. Reducing your DataLoader batch_size (e.g., from 64 to 16). 2. Invoking explicitly `import gc; gc.collect()` and `torch.cuda.empty_cache()`. 3. Deleting unused large tensors using `del tensor_name`.",
            metadata={"source": "cuda_troubleshooting.md", "topic": "oom_errors"}
        ),
        Document(
            page_content="To persistently mount Google Drive in Colab, execute: \nfrom google.colab import drive\ndrive.mount('/content/drive')\nAlways check paths relative to '/content/drive/MyDrive/' after mounting. Avoid frequent disconnects by keeping active data streams short.",
            metadata={"source": "drive_guide.md", "topic": "google_drive"}
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=35, length_function=len)
    split_docs = text_splitter.split_documents(kb_docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'} 
    )

    vector_store = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=DB_DIR,
        client_settings=Settings(anonymized_telemetry=False)
    )
    return vector_store

# --- 2. Session & Memory Manager ---
class SessionManager:
    def __init__(self, profile_path: str, log_path: str):
        self.profile_path = profile_path
        self.log_path = log_path
        self.short_term_memory: Dict[str, List[Dict[str, str]]] = {}

    def get_user_profile(self, user_id: str) -> str:
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r") as f:
                profiles = json.load(f)
                return profiles.get(user_id, {}).get("experience_level", "Beginner")
        return "Beginner"

    def update_short_term_memory(self, session_id: str, role: str, text: str, max_turns: int = 4):
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = []

        self.short_term_memory[session_id].append({"role": role, "text": text})
        if len(self.short_term_memory[session_id]) > (max_turns * 2):
            self.short_term_memory[session_id] = self.short_term_memory[session_id][-(max_turns * 2):]

    def get_formatted_history(self, session_id: str) -> str:
        history = self.short_term_memory.get(session_id, [])
        if not history:
            return "No previous conversation context."
        return "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in history])

    def log_interaction(self, session_id: str, query: str, context: list, response: str, latency: float):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query_length": len(query),
            "retrieved_sources": [doc.metadata.get("source") for doc in context],
            "response_length": len(response),
            "execution_latency_sec": round(latency, 3)
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

# --- 3. Prompt Builder ---
def prompt_template_builder(system_instructions: str, context: str, history: str, user_level: str, query: str) -> list:
    system_text = (
        f"{system_instructions}\n\n"
        f"[TARGET USER PROFILE]: {user_level}\n\n"
        f"[VERIFIED KNOWLEDGE CONTEXT]:\n{context}\n\n"
        f"[CONVERSATIONAL MEMORY]:\n{history}"
    )
    return [SystemMessage(content=system_text), HumanMessage(content=query)]

# --- 4. Global Instances ---
# Initialize DB and Manager when the module loads
##vector_db = initialize_knowledge_base()
session_manager = SessionManager(PROFILE_FILE, LOG_FILE)

# --- 5. Main RAG Pipeline ---
def run_colabbuddy_pipeline(user_id: str, session_id: str, user_query: str, hf_token: str) -> str:
    """Main execution function to be called from the Streamlit frontend."""
    start_time = time.time()

    # Configure LLM dynamically with the provided token
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="conversational",
        temperature=0.2,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token
    )
    chat_model = ChatHuggingFace(llm=llm_endpoint)

    # 1. Retrieval
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})
    retrieved_docs = retriever.invoke(user_query)
    context_str = "\n---\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Context Extraction
    user_level = session_manager.get_user_profile(user_id)
    chat_history = session_manager.get_formatted_history(session_id)

    # 3. Guardrails Formulation
    system_instructions = (
        "You are ColabBuddy, an elite technical support AI specializing in Google Colab Free Tier setups.\n"
        "Your goal is to resolve user errors safely using ONLY the provided verified knowledge context.\n"
        "Rules:\n"
        "1. If the solution cannot be derived from the context, explicitly say: 'I apologize, but I do not have enough verified documentation to answer this directly.'\n"
        "2. Do not hallucinate paths, limits, or configurations not listed in the context.\n"
        "3. Provide clean, efficient Python code blocks when answering code errors."
    )

    # 4. Construct Payload
    messages = prompt_template_builder(
        system_instructions=system_instructions,
        context=context_str if context_str else "No explicit context found.",
        history=chat_history,
        user_level=user_level,
        query=user_query
    )

    # 5. LLM Inference
    try:
        response = chat_model.invoke(messages)
        final_response = response.content.strip()
    except Exception as e:
        final_response = f"Fallback Triggered: {str(e)}"

    # 6. Memory & Metric Update (FIXED: Uses final_response)
    latency = time.time() - start_time
    session_manager.update_short_term_memory(session_id, "user", user_query)
    session_manager.update_short_term_memory(session_id, "assistant", final_response)
    session_manager.log_interaction(session_id, user_query, retrieved_docs, final_response, latency)

    return final_response