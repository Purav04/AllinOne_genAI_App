import sys, os
from src.logger import logger
from src.exception import CustomException    

from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings

from PIL import Image
import pytesseract
import speech_recognition as sr
import io
from gtts import gTTS

import streamlit as st

from src.text_input.qna import Question_Answer
from src.document_input.info_retriever import InfoRetriever as Document_InfoRetriever
from src.image_input.info_retriever import InfoRetriever as Image_InfoRetriever
from src.audio_input.translator import Translator

###

# create Page
st.set_page_config(page_title="Ask Question to LLM")
st.title("Ask Question to LLM")

# option to user what he wants to do
root_work = st.sidebar.selectbox("Choose Input Data Type:",
                                 ("Text", "Image", "Document", "Audio"))


if root_work == "Text":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Question and Answer"))

    if root_application_type == "Question and Answer":
        # call function and get response
        # response = text_QnA()
        response = Question_Answer().run()

        # write response
        st.write(response)
        # st.write_stream(LLM.invoke(input).content)  # not worked, bcz "streamlit.errors.StreamlitAPIException: st.write_stream expects a generator or stream-like object as input not <class 'str'>. Please use st.write instead for this data type."
        
        # for audio support # we won't build functon right now, when we work for audio then probably tomorrow.
        # if st.button("Speak Answer"):
        #     Speak(str(response))
        #     st.audio()

elif root_work == "Image":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Info Retrieved from Image"))

    if root_application_type == "Info Retrieved from Image":
        # call function and get response
        response = Image_InfoRetriever().run()

        # write response
        st.write(response)
        # st.write_stream(LLM.invoke(input).content)  # not worked, bcz "streamlit.errors.StreamlitAPIException: st.write_stream expects a generator or stream-like object as input not <class 'str'>. Please use st.write instead for this data type."
        # response = "Hello purav"
        
        # for audio support # we won't build functon right now, when we work for audio then probably tomorrow.
        # if st.button("Speak Answer"):
        #     Speak(str(response))
        #     st.audio()

elif root_work == "Document":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Info Retrieved from document"))
    
    if root_application_type == "Info Retrieved from document":
        # call function and get response
        # response = text_retriever_document()
        response = Document_InfoRetriever().run()

        # write response
        st.write(response)


elif root_work == "Audio":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Translation"))

    if root_application_type == "Translation":
        
            # text_output, audio_data = audio_translation()
            text_output, audio_data = Translator().run()
            st.write(f"Translation:\n\n{text_output}")
            
            if st.button("speak") and text_output:
                st.audio(audio_data, format='audio/mp3')


