import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_ser_model
from src.inference import predict_emotion_from_file
from src.config import EMOTIONS

def load_css():
    with open("demo/assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="demo/assets/icon.png",
)

load_css()

EMOTION_COLORS = {
    'angry': '#e74c3c', 'disgust': '#9b59b6', 'fear': '#e67e22',
    'happy': '#2ecc71', 'neutral': '#95a5a6', 'sad': '#3498db'
}

@st.cache_resource
def load_model():
    return load_ser_model()

def emotion_card(emotion, confidence=None):
    color = EMOTION_COLORS.get(emotion, '#666666')
    text = f"{emotion.upper()}{f' ({confidence:.1%})' if confidence else ''}"
    st.markdown(f'<div class="emotion-card" style="--emotion-color: {color}">{text}</div>', unsafe_allow_html=True)

def main():
    st.title("Speech Emotion Recognition")
    
    if (model := load_model()) is None:
        st.error("Failed to load model")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Audio")
        uploaded_file = st.file_uploader(" ", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], label_visibility="collapsed")
        
        if uploaded_file:
            st.audio(uploaded_file)
            
            if st.button("Analyze Emotion", type="primary", use_container_width=True):
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner("Analyzing..."):
                    result = predict_emotion_from_file(temp_path, model=model)
                
                os.remove(temp_path)
                
                if result:
                    with col2:
                        st.subheader("Result")
                        emotion_card(result['emotion'], result['confidence'])
                else:
                    st.error("Analysis failed")

    with col2:
        if not uploaded_file:
            st.subheader("Emotions")
            for emotion in EMOTIONS:
                emotion_card(emotion)

if __name__ == "__main__":
    main()