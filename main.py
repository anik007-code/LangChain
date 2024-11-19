import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser

llm = Ollama(model="llama2")

prompt_template = PromptTemplate(input_variables=["question"], template="{question}")
output_parser = StrOutputParser()
chain = LLMChain(llm=llm, prompt=prompt_template, output_key="response")

st.title('LangChain Chatbot with LLAMA2')

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
        response = chain({"question": user_input})["response"]
        st.session_state["history"].append({"user": user_input, "llm": response})

    st.experimental_rerun()
