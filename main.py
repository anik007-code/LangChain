import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv
import streamlit as st


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
            You are a helpful assistant. 
            Always respond directly and concisely to user queries. 
            Do not include introductory or concluding remarks such as 
            "Sure, I can help!" or "Let me know if you need anything else."
        """),
        ("user", "Question:{question}")
    ]
)

# Initialize LLM
llm = Ollama(model="llama2")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit
st.title('Langchain With LLAMA2 API')
if "history" not in st.session_state:
    st.session_state["history"] = []

def display_conversation(history):
    for entry in history:
        st.markdown(f"**User:** {entry['user']}")
        st.markdown(f"**LLM:** {entry['llm']}")

display_conversation(st.session_state["history"])

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Enter your message:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    with st.spinner("Waiting for the model's response..."):
        st.session_state["history"].append({"user": user_input, "llm": ""})
        placeholder = st.empty()
        response_parts = []

        conversation_history = "\n".join(
            f"User: {entry['user']}\nLLM: {entry['llm']}"
            for entry in st.session_state["history"][:-1]
        )
        enriched_question = f"Context:\n{conversation_history}\nQuestion: {user_input}"

        for chunk in chain.stream({"question": enriched_question}):
            response_parts.append(chunk)
            placeholder.markdown(f"**LLM:** {''.join(response_parts)}")
            time.sleep(0.1)

        full_response = ''.join(response_parts)
        st.session_state["history"][-1]["llm"] = full_response
        display_conversation(st.session_state["history"])
