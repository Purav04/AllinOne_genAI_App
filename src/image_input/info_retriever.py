from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

from PIL import Image
import pytesseract

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
        self.uploader_file = st.file_uploader("Uploade a Image", type=["png", "jpg", "jpeg"])

        # create input area
        self.input = st.text_area("Ask your Question: ")

        self.ask_button = st.button("Ask")
        
    def get_groq_image_retrieval(self, file, file_type, llm):
        # get embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            # get info from image
            image = Image.open(file)
            text_data = pytesseract.image_to_string(image).lower()
        except Exception as e:
            st.error(f"got error")
            CustomException(e, sys)
            return


        # Create a Chroma vector database and populate it
        chroma_db = Chroma.from_texts([text_data], collection_name="image_embedding", embedding=embeddings, persist_directory="Chroma_vectorestore")
        
        # creaton of prompt
        prompt_template = """
            Human: you have an invoice image data and give answer according to question.
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
        
        # Create a RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                chain_type="stuff",
                                                retriever=chroma_db.as_retriever(
                                                    search_type="similarity",
                                                    search_kwargs={"k": 3}
                                                ),
                                                return_source_documents=False,
                                                chain_type_kwargs={"prompt": prompt},
                                                )

        return qa_chain
        

    
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
            s
            if self.uploader_file:
                # add temp file
                temp_pdf_path = f"{__file__.rsplit('/', maxsplit=3)[0]}/temp/temp_{self.uploader_file.name}"
                os.makedirs(os.path.dirname(temp_pdf_path), exist_ok=True)
                with open(temp_pdf_path, "wb") as temp_file:
                    temp_file.write(self.uploader_file.read())
                
                # get retrieval
                retrieval = self.get_groq_image_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=LLM)
                
                if retrieval == None:
                    return ""

                if self.ask_button and len(self.input)>0:
                    response = retrieval.invoke({"query": self.input})
                    response = response["result"]

                    return response
        return ""

    def run(self):
        return self.ui()