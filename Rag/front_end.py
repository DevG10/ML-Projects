import streamlit as st
from backend import user_input

st.set_page_config(page_title="RAG Demo", page_icon="🤖")

st.title("RAG Demo using Streamlit and Google's Gemini Pro")

input = st.text_input("Enter your question here", key="user_input")

if st.button("Ask"):
    with st.spinner("Loading..."):
        answer = user_input(input)
    st.write("Answer: ", answer['output_text'])
