from langchain_groq import ChatGroq

import streamlit as st



class Question_Answer():
    def __init__(self):
        # selection of Framework
        self.root_framework = st.sidebar.selectbox("Select Framework",
                                            ("Groq", "Nvidia Nim"))
        
        # get required values from side bar
        self.api_key = st.sidebar.text_input("API Key", type="password")

        # create input area
        self.input = st.text_area("Ask your Question: ")

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key") 
    
    def ask_groq(self):
        # model
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))


        # Asking Question
        if st.button("Ask Question"):
            # add LLM model
            LLM = ChatGroq(model=llm_model, streaming=True, api_key=self.api_key)

            # Get response
            response = LLM.invoke(self.input)
            response = response.content

            return response

        return ""
    
    def ask_nvidia_nim(self):
        # get required values from side bar
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        ("meta/llama-3.3-70b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1", "google/gemma-2-9b-it"))
        
        return

    def ui(self):
        

        if self.root_framework == "Groq":
            response = self.ask_groq()

            return response

        elif self.root_framework == "Nvidia Nim":
            
            return "Work in Progress :)"

        return ""

    def run(self):
        return self.ui()