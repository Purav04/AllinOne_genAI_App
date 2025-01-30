from langchain_groq import ChatGroq

import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import io

class Translator():
    def __init__(self):
        self.lang =  {'af': 'Afrikaans', 'am': 'Amharic', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'bs': 'Bosnian', 'ca': 'Catalan', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fi': 'Finnish', 'fr': 'French', 'fr-CA': 'French (Canada)', 'gl': 'Galician', 'gu': 'Gujarati', 'ha': 'Hausa', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian', 'iw': 'Hebrew', 'ja': 'Japanese', 'jw': 'Javanese', 'km': 'Khmer', 'kn': 'Kannada', 'ko': 'Korean', 'la': 'Latin', 'lt': 'Lithuanian', 'lv': 'Latvian', 'ml': 'Malayalam', 'mr': 'Marathi', 'ms': 'Malay', 'my': 'Myanmar (Burmese)', 'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi (Gurmukhi)', 'pl': 'Polish', 'pt': 'Portuguese (Brazil)', 'pt-PT': 'Portuguese (Portugal)', 'ro': 'Romanian', 'ru': 'Russian', 'si': 'Sinhala', 'sk': 'Slovak', 'sq': 'Albanian', 'sr': 'Serbian', 'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai', 'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese', 'yue': 'Cantonese', 'zh-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Mandarin/Taiwan)', 'zh': 'Chinese (Mandarin)'}

        # selection of Framework
        self.root_framework = st.sidebar.selectbox("Select Framework",
                                            ("Groq", "Nvidia Nim"))

        # get required values from side bar
        # hf_api_key = st.sidebar.text_input("HuggingFace API Key", type="password") # not required
        self.api_key = st.sidebar.text_input("API Key", type="password")

        # check weather api key is written or not
        if self.api_key == "":
            st.sidebar.warning("Kindly Add API Key") 
            
        self.input_lang = st.selectbox("Input language: ", (i for i in self.lang.values()), placeholder="English")

        self.output_lang = st.selectbox("Output language: ", (i for i in self.lang.values()), placeholder="Spanish")

        self.audio_file = st.audio_input("Record a voice message")

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

    def ui(self):
        llm_model = st.sidebar.selectbox("Choose LLM model:", 
                                    ("llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"))

        if self.audio_file:
            transcript = self.get_text_from_audio(self.audio_file)
            if transcript and self.api_key:
                
                LLM = ChatGroq(model=llm_model, streaming=True, api_key=self.api_key)

                text_output = LLM.invoke(f"Translate this text {transcript} in {self.output_lang}. only provide translation").content

                audio_data = self.text_to_speech(text_output, [k for k,v in self.lang.items() if v==self.output_lang][-1])

                return text_output, audio_data    
        return "", ""

    def run(self):
        return self.ui()