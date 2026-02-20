import streamlit as st
from langchain_core.messages import HumanMessage
from agent import agent

st.set_page_config(page_title="Agent Demo", layout="centered")

st.title("🤖 File-Creating Agent (LangGraph + Groq)")

user_input = st.text_input("Ask the agent:")

if user_input:
    initial_state = {
        "messages": [
            HumanMessage(content=user_input)
        ]
    }

    result = agent.invoke(initial_state)

    final_message = result["messages"][-1]
    st.success(final_message.content)