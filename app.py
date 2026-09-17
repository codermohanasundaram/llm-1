import os
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from llm_client import ask, PRIMARY_MODEL, FRIENDLY_ERR_MSG


st.title("llm-1")
st.set_page_config(page_title="day1-llm", page_icon="💬")

api_key = os.environ.get("ANTHROPIC_API_KEY")

if not api_key:
    st.error("Set the ANTHROPIC_API_KEY environment variable (or put it in a .env file) and restart.")
    st.stop()
    
llm = ChatAnthropic(model=PRIMARY_MODEL, max_tokens=1024, api_key=api_key)

# Streamlit reruns this whole script on every interaction, so conversation
# history has to live in st.session_state to survive between messages.
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of LangChain message objects
    
# Render prior turns
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
# New input
if user_input := st.chat_input("Ask Claude something..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = ask(st.session_state.messages)  # timeout/retry/fallback handled inside
        if reply == FRIENDLY_ERR_MSG:
            st.warning(reply)  # visually distinct from a normal reply
        else:
            st.markdown(reply)
 
with st.sidebar:
    st.caption(f"Model: {PRIMARY_MODEL}")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
