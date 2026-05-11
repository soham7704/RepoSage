from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st

def get_direct_llm():
    return ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.3-70b-versatile"
    )
