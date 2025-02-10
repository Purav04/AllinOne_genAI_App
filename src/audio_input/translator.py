from langchain_groq import ChatGroq
import langchain_groq

import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.deepseek import DeepSeek
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io
from datetime import datetime

import sys, os
from src.exception import CustomException
from src.utils import create_cache_table, get_cache, get_table_size, set_cache, check_file_exist

class Translator():
    def __init__(self):
        self.list_models_groq = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"]
        self.list_models_llamaindex = {
            "OpenAI": ["o1", "o1-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
            "Anthropic": ["Claude 3.5 Haiku", "Claude 3 Opus", "Claude 3 Haiku"],
            "Gemini": ["gemini-2.0-flash-001", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"],
            "Mistral": ["Mistral-7B-Instruct-v0.2", "Mathstral-7B-v0.1"]
                            }

        self.lang =  {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fi': 'Finnish', 'fr': 'French', 'fr-CA': 'French (Canada)', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese', 'jw': 'Javanese', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lt': 'Lithuanian', 'lv': 'Latvian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi (Gurmukhi)', 'pl': 'Polish', 'pt': 'Portuguese (Brazil)', 'pt-PT': 'Portuguese (Portugal)', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'yue': 'Cantonese', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Mandarin/Taiwan)', 'zh': 'Chinese (Mandarin)'}

        # selection of Framework
        self.root_framework = st.sidebar.selectbox("Select Framework",
                                            ("Groq", "Llama-Index"))

        # get required values from side bar
        # hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password") # not required
        self.api_key = st.sidebar.text_input("API Key", type="password")

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key") 
            
        self.input_lang = st.selectbox("Input language: ", (i for i in self.lang.values()), placeholder="English")

        self.output_lang = st.selectbox("Output language: ", (i for i in self.lang.values()), placeholder="Spanish")

        self.audio_file = st.audio_input("Record a voice message")

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
                create_cache_table(database_file_name=f"cache_db/{self.db_file_name}")
        

    def text_to_speech(self, text, lang):
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


    def get_text_from_audio(self, audio_file):
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
        
    def ask_groq(self):
        if self.audio_file:
            transcript = self.get_text_from_audio(self.audio_file)
            if transcript and self.api_key:
                cached_response = get_cache(key=transcript, database_file_name=self.db_file_name)
                if cached_response:
                    response = "From Cache:\n\n\n" + str(cached_response)
                    return response
                try:
                    LLM = ChatGroq(model=self.llm_model, streaming=True, api_key=self.api_key, 
                                   temperature=self.tempereature, max_tokens=self.max_tokens)
                except Exception as e:
                    st.error(f"got error")
                    CustomException(e, sys)
                    return "",""
                    
                
                text_output = LLM.invoke(f"Translate this text {transcript} in {self.output_lang}. only provide translation").content
                set_cache(key=transcript, value=text_output, database_file_name=self.db_file_name)
                
                audio_data = self.text_to_speech(text_output, [k for k,v in self.lang.items() if v==self.output_lang][-1])

                return text_output, audio_data    
        return "", ""
    
    def ask_llamaindex(self):
        if self.audio_file:
            transcript = self.get_text_from_audio(self.audio_file)
            if transcript and self.api_key:
                
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
                    

                text_output = llm.invoke(f"Translate this text {transcript} in {self.output_lang}. only provide translation").content
                set_cache(key=transcript, value=text_output, database_file_name=self.db_file_name)
                
                audio_data = self.text_to_speech(text_output, [k for k,v in self.lang.items() if v==self.output_lang][-1])

                return text_output, audio_data    
        return "", ""

    def ui(self):
        if self.root_framework == "Groq":
            return self.ask_groq()
        elif self.root_framework == "Llama-Index":
            pass

        

    def run(self):
        return self.ui()