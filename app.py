import streamlit as st
import base64
import os
import shutil
from github_utils import fetch_repo_files, extract_github_info
from loader import load_repo_files, chunk_documents
from vectorstore_utils import get_vectorstore, load_vectorstore
from qa_utils import ask_question
from qa_chain import get_direct_llm
from dotenv import load_dotenv

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    if not groq_api_key:
        raise ValueError("Empty key")
except Exception:
    st.error("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
    st.stop()

# Load GitHub icon
try:
    with open("icon-github.png", "rb") as img_file:
        github_icon_base64 = base64.b64encode(img_file.read()).decode()
except FileNotFoundError:
    github_icon_base64 = ""
    st.error("Error: icon-github.png not found. Please ensure it's in the same directory.")

# Page config
st.set_page_config(page_title="RepoSage", layout="wide")

st.markdown("""
<style>
.open-sidebar-hint {
    position: fixed;
    top: 60px;
    left: 10px;
    z-index: 10001;
    background: #232323ee;
    color: #fffbe7;
    padding: 7px 16px 7px 16px;
    border-radius: 15px;
    font-size: 15px;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.13);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
    border: 2px solid #ffc107;
}
@media (min-width: 768px) {
  .open-sidebar-hint { display: none; }
}
</style>
<script>
function clickSidebarBtn() {
    var sideBtn = window.parent.document.querySelector('[data-testid="collapsedControl"]');
    if (sideBtn){ sideBtn.click(); }
}
</script>

<div class="open-sidebar-hint" onclick="clickSidebarBtn()">
    👆🏻 Tap to open sidebar
</div>
""", unsafe_allow_html=True)

# Custom CSS styles
st.markdown(f"""
    <style>
        .stChatMessage.user {{ text-align: right; }}
        .stChatMessage.assistant {{ text-align: left; }}
        .message-label {{ font-weight: bold; font-size: 18px; }}
        .message-content {{ font-size: 15px; margin-top: 5px; }}
        .custom-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .custom-logo img {{ height: 30px; }}
        .custom-logo span {{ font-size: 24px; font-weight: bold; }}
        .disclaimer-box {{
            background-color: #fff3cd;
            color: #856404;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 15px;
            border: 1px solid #ffeeba;
            margin-top: 10px;
            line-height: 1.5em;
            margin-bottom: 16px;
        }}
        .academic-note {{
            background-color: #fff3cd;
            color: #856404;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 15px;
            border: 1px solid #ffeeba;
            margin-top: auto;
        }}
        .header-logo {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .header-logo img {{ height: 40px; }}
        .header-logo h1 {{ margin: 0; }}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar UI
with st.sidebar:
    if github_icon_base64:
        st.markdown(f"""
            <div class="custom-logo">
                <img src="data:image/png;base64,{github_icon_base64}" />
                <span>RepoSage</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🔗 Upload GitHub Repo Link")

    with st.form(key="repo_form"):
        st.markdown("Paste GitHub URL")
        github_url = st.text_input("", label_visibility="collapsed", key="github_url_input")
        process_button = st.form_submit_button("🔍 Analyze Repository")

    st.markdown("""
        <div class="disclaimer-box">
            🔐 <i><b>Disclaimer:</b><br>No data is stored. All processing is done locally and securely.</i>
        </div>
    """, unsafe_allow_html=True)

    # Process repository
    if github_url and process_button:
        if os.path.exists("faiss_index"):
            try:
                shutil.rmtree("faiss_index")
            except PermissionError as e:
                st.error(f"Error: Could not remove faiss_index directory: {e}")
                st.stop()

        with st.spinner("Analyzing your repository..."):
            try:
                owner, repo = extract_github_info(github_url)
                files = fetch_repo_files(owner, repo)
                docs = load_repo_files(files)
                chunks = chunk_documents(docs)

                vectorstore = get_vectorstore(chunks)
                st.session_state.retriever = load_vectorstore(vectorstore)
                st.session_state.llm = get_direct_llm()
                st.session_state.chat_history = []

                st.success("✅ Repository indexed and ready to query!")
            except Exception as e:
                st.error(f"❌ Error processing repository: {e}")
                st.stop()
    
    # Clear chat button
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

    # 📌 Academic disclaimer at the bottom
    if "academic_note_rendered" not in st.session_state:
        st.session_state.academic_note_rendered = True
        st.markdown("""
                    <div class="academic-note">
                    ⚠️ <i><b>Note:</b><br>
                    This project is for academic enhancement and learning purposes only.
                    Recommended for use with small to medium-sized repositories.</i>
                    </div>
                    """, unsafe_allow_html=True)

# Header
if github_icon_base64:
    st.markdown(f"""
        <div class="header-logo">
            <img src="data:image/png;base64,{github_icon_base64}" />
            <h1 style='display:inline-block;'>RepoSage</h1>
        </div>
    """, unsafe_allow_html=True)

st.markdown("## Your Personal GitHub Repository Assistant")
st.markdown("### 🧠 Ask about your Repo")

# Chat section
if "retriever" in st.session_state:

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            label = "🧑‍💼 You:" if msg["role"] == "user" else "📘 RepoSage:"
            st.markdown(f"""
                <div class='message-label'>{label}</div>
                <div class='message-content'>{msg['content']}</div>
            """, unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Type your question here...", key="chat_input")

    if user_question:
        with st.chat_message("user"):
            st.markdown(f"""
                <div class='message-label'>🧑‍💼 You:</div>
                <div class='message-content'>{user_question}</div>
            """, unsafe_allow_html=True)

        with st.spinner("Finding your answer..."):
            try:
                result = ask_question(
                    st.session_state.retriever,
                    user_question,
                    st.session_state.chat_history
                )

                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": result})

                with st.chat_message("assistant"):
                    st.markdown(f"""
                        <div class='message-label'>📘 RepoSage:</div>
                        <div class='message-content'>{result}</div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error answering question: {e}")
else:
    st.info("📥 Paste a valid GitHub repo URL and click Analyze to begin.")
