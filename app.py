import sys, os
from src.logger import logging
from src.exception import CustomException    

from langchain_groq import ChatGroq
import groq

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# create Page
st.set_page_config(page_title="Ask Question to LLM")
st.title("Ask Question to LLM")

# create side bar
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")

# create input area
input = st.text_input("Ask your Question: ")

# check weather api key is written or not
if groq_api_key == "":
    st.sidebar.write("Kindly Add API Key") 
else:
    st.sidebar.write("") 

# Asking Question
if st.button("Ask Question"):
        # add LLM model
        LLM = ChatGroq(model="Llama3-8b-8192", streaming=True, api_key=groq_api_key)

        # Get response
        response = LLM.invoke(input)
        st.write(response.content)
        