import sys, os
from src.logger import logging
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
# from gtts import gTTS
# import tempfile

###

# def Speak(text_to_speak):
#     # Create a gTTS object
#     tts = gTTS(text=text_to_speak, lang='en')

#     # Save the audio file temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
#         tmpfile.close()
#         tts.save(tmpfile.name)

#         # Play the audio using Streamlit's audio player
#         st.audio(tmpfile.name, format="audio/mp3")

def get_image_retrieval(file, file_type, llm):
    # get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # get info from image
    image = Image.open(file)
    text_data = pytesseract.image_to_string(image).lower()


    # Create a Chroma vector database and populate it
    chroma_db = Chroma.from_texts([text_data], embedding=embeddings)
    
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


def get_document_retrieval(file, file_type, llm):
    # get embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # load pdf
    loader = PyPDFLoader(file)
    documents = loader.load()

    # get text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print(f"document docs- type:{type(docs)} \n number_of_elements:{len(docs)} \n type_of_first_element:{type(docs[0])} \n data:{docs}")

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
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
    
    return qa

# Function to transcribe audio using SpeechRecognition with the 'google' engine
def get_audio_text(audio_file):
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(audio_data)
        return text

    except sr.UnknownValueError as e:
        st.warning("Google Speech Recognition could not understand audio")
        print(f"error: {e}")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    
def text_to_speech(text, lang='en'):
    """
    Converts text to speech using gTTS.

    Args:
        text: The text to be converted to speech.
        lang: The language code for the output speech (default: 'en').

    Returns:
        A base64 encoded string of the audio data.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        
        return audio_file
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

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
        retrieval = get_document_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=LLM)

        # delete temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


        # Asking Question
        if st.button("Ask Question") and input:
            
            response = retrieval({"query": input})
            response = response["result"]
    
    return response

def text_retriever_image():
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
    uploader_file = st.file_uploader("Uploade a Document", type=["png", "jpg", "jpeg"])

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
        retrieval = get_image_retrieval(file=temp_pdf_path, file_type=temp_pdf_path.rsplit(".")[-1], llm=LLM)

        # delete temp file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


        # Asking Question
        if st.button("Ask Question") and input:
            
            response = retrieval.invoke({"query": input})
            response = response["result"]

    return response

def audio_translation():
    lang =  {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fi': 'Finnish', 'fr': 'French', 'fr-CA': 'French (Canada)', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese', 'jw': 'Javanese', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lt': 'Lithuanian', 'lv': 'Latvian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi (Gurmukhi)', 'pl': 'Polish', 'pt': 'Portuguese (Brazil)', 'pt-PT': 'Portuguese (Portugal)', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'yue': 'Cantonese', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Mandarin/Taiwan)', 'zh': 'Chinese (Mandarin)'}

    # selection of Framework
    root_framework = st.sidebar.selectbox("Select Framework",
                                        ("Groq", "Nvidia Nim"))
    
    # get required values from side bar
    # hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password") # not required
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
    llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                    ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))

    input_lang = st.selectbox("Input language: ", (i for i in lang.values()), placeholder="English")

    output_lang = st.selectbox("Output language: ", (i for i in lang.values()), placeholder="Spanish")

    audio_file = st.audio_input("Record a voice message")

    if audio_file:
        transcript = get_audio_text(audio_file)
        if transcript:
            
            LLM = ChatGroq(model=llm_model, streaming=True, api_key=groq_api_key)

            text_output = LLM.invoke(f"Translate this text {transcript} in {output_lang}. only provide translation").content

            audio_data = text_to_speech(text_output, [k for k,v in lang.items() if v==output_lang][-1])

            return text_output, audio_data    
    return "", ""



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
        
        # for audio support # we won't build functon right now, when we work for audio then probably tomorrow.
        # if st.button("Speak Answer"):
        #     Speak(str(response))
        #     st.audio()
            
    elif root_application_type == "Info Retrieved from document":
        # call function and get response
        response = text_retriever_document()

        # write response
        st.write(response)

elif root_work == "Image":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Info Retrieved from Image"))

    if root_application_type == "Info Retrieved from Image":
        # call function and get response
        response = text_retriever_image()

        # write response
        st.write(response)
        # st.write_stream(LLM.invoke(input).content)  # not worked, bcz "streamlit.errors.StreamlitAPIException: st.write_stream expects a generator or stream-like object as input not <class 'str'>. Please use st.write instead for this data type."
        # response = "Hello purav"
        
        # for audio support # we won't build functon right now, when we work for audio then probably tomorrow.
        # if st.button("Speak Answer"):
        #     Speak(str(response))
        #     st.audio()

elif root_work == "Audio":
    root_application_type = st.sidebar.selectbox("Choose Application type:",
                                                 ("Translation"))

    if root_application_type == "Translation":
        
            text_output, audio_data = audio_translation()
            st.write(text_output)
            
            if st.button("speak") and text_output:
                st.audio(audio_data, format='audio/mp3')




