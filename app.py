##  Application file ##

import streamlit as st
import os
import time
from config import CONFIG

# Import the functional modules built in Section 2
from backend_core import run_colabbuddy_pipeline, session_manager

# --- Page Setup & Accessibility Configuration ---
st.set_page_config(page_title="ColabBuddy Support AI", page_icon="🤖", layout="wide")

# --- UI Sidebar: Guidance, Capabilities & Disclaimers ---
with st.sidebar:
    st.title("🤖 ColabBuddy Support")
    st.markdown("### Free-Tier Assistant")
    st.info(
        "**Capabilities:**\n"
        "- Troubleshooting RAM/VRAM CUDA crashes.\n"
        "- Working around the 12.5-hour session execution limits.\n"
        "- Google Drive storage connection assistance."
    )
    
    st.markdown("### ⚠️ Limitations & Disclaimer")
    st.warning(
        "ColabBuddy is a prototype support system. It may occasionally "
        "hallucinate code syntax. It cannot access your actual Google account or active notebooks."
    )
    
    st.markdown("### 💡 Try These Prompts:")
    st.code("My code crashed with 'CUDA out of memory'. How do I fix it?")
    st.code("How long can I run my script before Colab disconnects?")

# --- Main Interface Header ---
st.title("Google Colab Technical Support Chatbot")
st.caption("Empowering hobbyists, students, and researchers with instant runtime debugging.")
st.divider()

# --- Initialize Application State Management ---
USER_ID = "tester_user"
SESSION_ID = "active_streamlit_session"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History Window ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input Execution Window ---
if user_query := st.chat_input("Paste your Colab error message or question here..."):
    # 1. Display User Turn
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Display Chatbot Turn with dynamic loading state
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        with st.spinner("Analyzing knowledge framework / logs..."):
            # Core processing execution
            bot_response = run_colabbuddy_pipeline(USER_ID, SESSION_ID, user_query)
            
        # Simulated streaming response effect for better UI usability
        full_response = ""
        for chunk in bot_response.split(" "):
            full_response += chunk + " "
            time.sleep(0.04)
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
    # Append response to memory sequence
    st.session_state.messages.append({"role": "assistant", "content": full_response})