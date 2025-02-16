from langchain_groq import ChatGroq
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI

import streamlit as st
from datetime import datetime
from dataclasses import dataclass

import sys, os
from src.exception import CustomException
from src.utils import create_cache_table, get_cache, get_table_size, set_cache, check_file_exist

@dataclass
class Question_Answer():
    def __init__(self):
        self.list_models_groq = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        self.list_models_llamaindex = {
            "OpenAI": ["o1", "o1-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
            "Anthropic": ["Claude 3.5 Haiku", "Claude 3 Opus", "Claude 3 Haiku"],
            "Gemini": ["gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
            "Mistral": ["Mistral-7B-Instruct-v0.2", "Mathstral-7B-v0.1"]
                            }
        
        # selection of Framework
        self.root_framework = st.sidebar.selectbox("Select Framework",
                                            ("Groq", "Llama-Index"))
        
        # get required values from side bar
        self.api_key = st.sidebar.text_input("API Key", type="password")

        # create input area
        self.input = st.text_area("Ask your Question: ")

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key")

        if self.root_framework == "Groq":
            self.llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        (i for i in self.list_models_groq))
        elif self.root_framework == "Llama-Index":
            self.llm_models_provider = st.sidebar.selectbox("Select LLM Model Provider:",
                                                   (i for i in self.list_models_llamaindex.keys()))
        
            self.llm_model = st.sidebar.selectbox("Select llm model:",
                                            (i for i in self.list_models_llamaindex[self.llm_models_provider]))
        

        self.tempereature = st.sidebar.slider("Temperature", 0.0, 1.0, step=0.1, value=0.5)
        self.max_tokens = st.sidebar.slider("Max Number of Tokens", 128, 1024, step=128, value=1024)
 

        with st.sidebar.expander("Advanced Settings"):
            self.allow_caching = st.radio(
                    "Allow Caching:",
                    ("Yes", "No")
                )
            
        if self.allow_caching == "Yes":
            self.db_file_name = f"cache_db/cache_{datetime.now().strftime('%Y-%m-%d')}.db"
            if not check_file_exist(file_name=self.db_file_name, path=os.path.join(os.getcwd(), "cache_db") ):
                create_cache_table(database_file_name=self.db_file_name)

    
    def ask_groq(self):
        
        if self.api_key:
            cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
            if cached_response:
                response = "From Cache:\n\n\n" + str(cached_response)
                return response
            else:
                try:
                    LLM = ChatGroq(model=self.llm_model, streaming=True, api_key=self.api_key, 
                                    temperature=self.tempereature, max_tokens=self.max_tokens)
                except Exception as e:
                    st.error(f"got error")
                    CustomException(e, sys)
                    return ""

                # Get response
                response = LLM.invoke(self.input)
                response = response.content
                set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                return response

        return ""
    
    def ask_nvidia_nim(self):
        # get required values from side bar
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        ("meta/llama-3.3-70b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1", "google/gemma-2-9b-it"))
        
        return

    def ask_llamaindex(self):

        if self.api_key:
            try:
                if self.llm_models_provider == "OpenAI":
                    cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                    if cached_response:
                        response = "From Cache:\n\n" + str(cached_response)
                    else:
                        llm = OpenAI(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        response = "for OpenAI Models, Work is in Progress"
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                    
                elif self.llm_models_provider == "DeepSeek":
                    cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                    if cached_response:
                        response = "From Cache:\n\n" + str(cached_response)
                    else:
                        llm = DeepSeek(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        response = llm.complete(self.input)
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)

                elif self.llm_models_provider == "Anthropic":
                    cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                    if cached_response:
                        response = "From Cache:\n\n" + str(cached_response)
                    else:
                        llm = Anthropic(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        response = llm.stream_complete(self.input)
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                   
                elif self.llm_models_provider == "Gemini":
                    cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                    if cached_response:
                        response = "From Cache:\n\n" + str(cached_response)
                    else:
                        llm = Gemini(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        response = llm.complete(self.input)
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                    

                elif self.llm_models_provider == "Mistral":
                    cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                    if cached_response:
                        response = "From Cache:\n\n" + str(cached_response)
                    else:
                        llm = MistralAI(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        response = llm.complete(self.input)
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                    

            except Exception as e:
                st.error(f"got error while connecting model with API.Kindly check it again.")
                CustomException(e, sys)
                return ""

            return response
        return ""

    def ui(self):
        if self.root_framework == "Groq":
            response = self.ask_groq()

            return response

        elif self.root_framework == "Llama-Index":
            response = self.ask_llamaindex()
            return response

        return ""

    def run(self):
        return self.ui()