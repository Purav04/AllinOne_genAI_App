import sys
from src.logger import logging
from src.exception import CustomException    
import streamlit as st

st.set_page_config(page_title="Ask Question to LLM")
st.title("Ask Question to LLM")

input = st.text_input("Ask your Question: ", key="input")

