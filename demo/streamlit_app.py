import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_ser_model
from src.inference import predict_emotion_from_file
from src.config import EMOTIONS

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎭")

EMOTION_COLORS = {
    'angry': '#ff4444', 'disgust': '#9b59b6', 'fear': '#ff8c00',
    'happy': '#00d084', 'neutral': '#6c757d', 'sad': '#3498db'
}

@st.cache_resource
def load_model():
    return load_ser_model()

def emotion_card(emotion, confidence=None):
    color = EMOTION_COLORS.get(emotion, '#666666')
    text = f"{emotion.upper()}{f' ({confidence:.1%})' if confidence else ''}"
    st.markdown(f"""
        <div style='background-color: {color}; color: white; padding: 1rem; 
                    border-radius: 8px; text-align: center; margin: 0.5rem 0;'>
            {text}
        </div>
    """, unsafe_allow_html=True)

def main():
    st.title("Speech Emotion Recognition")
    
    if (model := load_model()) is None:
        st.error("Failed to load model.")
        st.stop()
    
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3'])
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Analyze", type="primary"):
            with st.spinner("Processing..."):
                result = predict_emotion_from_file(temp_path, model=model)
            
            os.remove(temp_path)
            
            if result:
                emotion_card(result['emotion'], result['confidence'])
            else:
                st.error("Analysis failed")
    else:
        st.info("Upload an audio file to analyze emotion")
        cols = st.columns(3)
        for i, emotion in enumerate(EMOTIONS):
            with cols[i % 3]:
                emotion_card(emotion)

if __name__ == "__main__":
    main()