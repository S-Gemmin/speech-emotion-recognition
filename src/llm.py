import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from src.config import LLM_TYPE, MAX_TOKENS, TEMPERATURE

def create_llm_client():
    try:
        load_dotenv()
        llm_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        return llm_client
    except Exception as e:
        print(e)
        return None 
    
def generate_response(transcript, emotion_scores, model):
    prompt = f'''
        The user is speaking in a {emotion_scores} tone. They said: '{transcript}'. 
        Generate a very short, appropriate response. You can be emotional 
        and empathetic in your reply. 
    '''
    
    try:
        chat_completion = model.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model=LLM_TYPE,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        return 'I am down at the moment. Sorry about that!'
