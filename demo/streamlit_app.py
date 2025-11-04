import streamlit as st
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import load_ser_model
from src.inference import predict_emotion_from_file

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="demo/assets/logo.png")

@st.cache_resource
def load_model(): 
    return load_ser_model()

def main():
    st.title("Speech Emotion Recognition")
    if (model := load_model()) is None: 
        return st.error("Model failed to load")
    
    if uploaded_file := st.file_uploader("Upload audio", type=['wav', 'mp3', 'flac']):
        st.audio(uploaded_file)

        if st.button("Analyze"):
            with open(f"temp_{uploaded_file.name}", "wb") as f: 
                f.write(uploaded_file.getbuffer())
            
            if result := predict_emotion_from_file(f"temp_{uploaded_file.name}", model=model):
                st.success(f"Emotion: {result['emotion'].upper()} ({result['confidence']:.1%})")

if __name__ == "__main__": main()