import sys, os
from src.logger import logging
from src.exception import CustomException    

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings


import streamlit as st
from gtts import gTTS
import tempfile

###

def Speak(text_to_speak):
    # Create a gTTS object
    tts = gTTS(text=text_to_speak, lang='en')

    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
        tmpfile.close()
        tts.save(tmpfile.name)

        # Play the audio using Streamlit's audio player
        st.audio(tmpfile.name, format="audio/mp3")

def get_retrieval(pdf_file, llm):
    # get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # load pdf
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()

    
    # get text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # get vector store
    vector_store = Chroma.from_documents(collection_name="pdf_embedding", embedding=embeddings, documents=docs, persist_directory="chroma_vectores")

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
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    return qa


def text_QnA():
    # selection of Framework
    root_framework = st.sidebar.selectbox("Select Framework",
                                          ("Groq", "Nvidia Nim"))
    response = ""
    if root_framework == "Groq":
        # get required values from side bar
        groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))

        # create input area
        input = st.text_area("Ask your Question: ")

        # check weather api key is written or not
        if groq_api_key == "":
            st.sidebar.warning("Kindly Add API Key") 

        # Asking Question
        if st.button("Ask Question"):
            # add LLM model
            LLM = ChatGroq(model=llm_model, streaming=True, api_key=groq_api_key)

            # Get response
            response = LLM.invoke(input)
            response = response.content

        return response
    elif root_framework == "Nvidia Nim":
        # get required values from side bar
        nvidia_nim_api_key = st.sidebar.text_input("Nvidia Nim API Key", type="password")
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                        ("meta/llama-3.3-70b-instruct", "mistralai/mixtral-8x22b-instruct-v0.1", "google/gemma-2-9b-it"))
        
        # create input area
        input = st.text_area("Ask your Question: ")

        # check weather api key is written or not
        if nvidia_nim_api_key == "":
            st.sidebar.warning("Kindly Add API Key") 

        return "Working in Progress :)"

def text_retriever_document():
    # selection of Framework
    root_framework = st.sidebar.selectbox("Select Framework",
                                          ("Groq", "Nvidia Nim"))
    
    # get required values from side bar
    # hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password") # not required
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                    ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))

    # check weather api key is written or not
    if groq_api_key == "":
        st.sidebar.warning("Kindly Add API Key") 
    response = ""

    # file uploader
    uploader_file = st.file_uploader("Uploade a Document", type=["pdf", "docs"])

    if uploader_file:
        # add temp file
        temp_pdf_path = f"temp_{uploader_file.name}"
        with open(temp_pdf_path, "wb") as temp_file:
            temp_file.write(uploader_file.read())

        # create input area
        input = st.text_area("Ask your Question: ")

        # add LLM model
        LLM = ChatGroq(model=llm_model, streaming=True, api_key=groq_api_key)

        # get retrieval
        retrieval = get_retrieval(pdf_file=temp_pdf_path, llm=LLM)

        # delete temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


        # Asking Question
        if st.button("Ask Question") and input:
            
            response = retrieval({"query": input})
            response = response["result"]
    
    return response



# create Page
st.set_page_config(page_title="Ask Question to LLM")
st.title("Ask Question to LLM")

# option to user what he wants to do
root_work = st.sidebar.selectbox("Choose what operation you want to do:",
                                 ("Text", "Image", "Audio"))


if root_work == "Text":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Question and Answer", "Info Retrieved from document"))

    if root_application_type == "Question and Answer":
        # call function and get response
        response = text_QnA()

        # write response
        st.write(response)
        # st.write_stream(LLM.invoke(input).content)  # not worked, bcz "streamlit.errors.StreamlitAPIException: st.write_stream expects a generator or stream-like object as input not <class 'str'>. Please use st.write instead for this data type."
        if response and st.button("Speak Text"):
            Speak(response)
            
    elif root_application_type == "Info Retrieved from document":
        # call function and get response
        response = text_retriever_document()

        # write response
        st.write(response)
else:
    st.write("Working in Progress :)")
