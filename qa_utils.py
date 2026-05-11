import streamlit as st
from langchain_community.chains import ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

def ask_question(retriever, question, chat_history):
    if "llm" not in st.session_state or st.session_state.llm is None:
        raise ValueError("LLM model not loaded. Please analyze a repository first.")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=st.session_state.llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False
    )

    result = qa_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

    return result["answer"]
