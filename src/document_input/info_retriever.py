from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI

import streamlit as st
import os
from datetime import datetime
from dataclasses import dataclass

import sys
from src.exception import CustomException
from src.utils import create_cache_table, get_cache, get_table_size, set_cache, check_file_exist

@dataclass
class InfoRetriever():
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

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key") 
        
        # file uploader
        self.uploader_file = st.file_uploader("Uploade a Document", type=["pdf"])

        # create input area
        self.input = st.text_area("Ask your Question: ")

        # get text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

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


    def get_document_retrieval(self, file, file_type, llm):
        # embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


        # load pdf
        loader = PyPDFLoader(file)
        documents = loader.load()

        # convert text into splited text
        docs = self.text_splitter.split_documents(documents)

        # get vector store
        vector_store = Chroma.from_documents(collection_name="pdf_embedding", embedding=embeddings, documents=docs, persist_directory="Chroma_vectorestore")

        # create prompt template
        prompt_template = """
            Human: Give the precisely and in the limit of 250 words.
            <context>
            {context}
            </context>

            Question: {question}

            Assistant:
            """
        prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        
        # getting retrieval
        qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}
                ),
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt}
            )
        
        return qa
    
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

                if self.uploader_file:
                    # add temp file
                    temp_pdf_path = f"{__file__.rsplit('/', maxsplit=3)[0]}/temp/temp_{self.uploader_file.name}"
                    os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
                    with open(temp_pdf_path, "wb") as temp_file:
                        temp_file.write(self.uploader_file.read())
                    
                    # get retrieval
                    retrieval = self.get_document_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=LLM)
                    
                    if len(self.input)>0:
                        response = retrieval.invoke({"query": self.input})
                        response = response["result"]
                        set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                        return response
        return ""

    def ask_llamaindex(self):
        if self.api_key:
            try:
                cached_response = get_cache(key=self.input, database_file_name=self.db_file_name)
                if cached_response:
                    response = "From Cache:\n\n" + str(cached_response)
                    return response
                if self.llm_models_provider == "OpenAI":
                    llm = OpenAI(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        
                elif self.llm_models_provider == "DeepSeek":
                    llm = DeepSeek(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        
                elif self.llm_models_provider == "Anthropic":
                    llm = Anthropic(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        
                elif self.llm_models_provider == "Gemini":
                    llm = Gemini(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        
                elif self.llm_models_provider == "Mistral":
                    llm = MistralAI(model=self.llm_model, api_key=self.api_key, temperature=self.tempereature, max_tokens=self.max_tokens)
                        

            except Exception as e:
                st.error(f"got error while connecting model with API.Kindly check it again.")
                CustomException(e, sys)
                return ""

            if self.uploader_file:
                # add temp file
                temp_pdf_path = f"{__file__.rsplit('/', maxsplit=3)[0]}/temp/temp_{self.uploader_file.name}"
                os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
                with open(temp_pdf_path, "wb") as temp_file:
                    temp_file.write(self.uploader_file.read())
                
                # get retrieval
                retrieval = self.get_document_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=llm)
                
                if len(self.input)>0:
                    response = retrieval.invoke({"query": self.input})
                    response = response["result"]
                    set_cache(key=self.input, value=response, database_file_name=self.db_file_name)
                    return response
        return ""

    def ui(self):
        if self.root_framework == "Groq":
            return self.ask_groq()
        elif self.root_framework == "Llama-Index":
            return self.ask_llamaindex()

    def run(self):
        return self.ui()
