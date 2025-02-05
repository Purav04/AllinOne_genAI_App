from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

import streamlit as st
import os

import sys
from src.exception import CustomException

class InfoRetriever():
    def __init__(self):
        # selection of Framework
        self.root_framework = st.sidebar.selectbox("Select Framework",
                                            ("Groq", "Nvidia Nim"))
        
        # get required values from side bar
        self.api_key = st.sidebar.text_input("API Key", type="password")

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key") 
        
        # file uploader
        self.uploader_file = st.file_uploader("Uploade a Document", type=["pdf"])

        # create input area
        self.input = st.text_area("Ask your Question: ")

        self.ask_button = st.button("Ask")
        
        # get text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def get_groq_document_retrieval(self, file, file_type, llm):
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

    def ui(self):
        # select LLM models
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                    ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))
        tempereature = st.sidebar.slider("Temperature", 0.0, 1.0, step=0.1, value=0.5)
        max_tokens = st.sidebar.slider("Max Number of Tokens", 128, 1024, step=128, value=1024)

        if self.api_key:
            try:
                LLM = ChatGroq(model=llm_model, streaming=True, api_key=self.api_key, 
                                   temperature=tempereature, max_tokens=max_tokens)
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
                retrieval = self.get_groq_document_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=LLM)
                
                if self.ask_button and len(self.input)>0:
                    response = retrieval.invoke({"query": self.input})
                    response = response["result"]

                    return response
        return ""


    def run(self):
        return self.ui()
