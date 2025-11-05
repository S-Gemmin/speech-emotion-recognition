import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_ser_model
from src.inference import predict_emotion_from_file
from src.config import SUPPORTED_FORMATS
from src.stt import load_stt_model, speech_to_text
from src.llm import create_llm_client, generate_response

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="assets/icon.png",
)

@st.cache_resource
def load_models():
    return {
        'ser': load_ser_model(),
        'stt': load_stt_model(), 
        'llm': create_llm_client()
    }

def get_response(path, models):
    temp_path = f"temp_{path.name}"
    with open(temp_path, "wb") as f:
        f.write(path.getbuffer())
            
    with st.spinner("Processing..."):
        emotion_result = predict_emotion_from_file(temp_path, model=models['ser'])
        transcript = speech_to_text(temp_path, model=models['stt'])
        llm_response = generate_response(transcript, emotion_result['emotion'], models['llm'])

    os.remove(temp_path)

    return {
        'emotion_result': emotion_result,
        'transcript': transcript,
        'llm_response': llm_response
    }

def main():
    st.title("Speech Emotion Recognition + LLM")

    st.markdown(
    """
    Combines speech emotion recognition with large language models to create 
    emotionally intelligent responses. Upload any audio file to analyze
    vocal emotions and receive context-aware responses. Here are the supported
    emotions:
    - Angry
    - Happy
    - Sad
    - Neutral
    - Fearful
    - Disgusted
    """
    )
    
    models = load_models()
    if models is None:
        st.error("Failed to load models")
        st.stop()

    uploaded_file = st.file_uploader("Upload Audio", type=SUPPORTED_FORMATS)
    st.audio(uploaded_file)
    response = {'emotion_result': None, 'transcript': None, 'llm_response': None}

    if st.button("Analyze Audio", type="secondary", use_container_width=True) and uploaded_file is not None:
        response = get_response(uploaded_file, models)

    st.subheader("Response")

    st.code(
    f"""
    Emotion: {response['emotion_result']['emotion'] if response['emotion_result'] else 'N/A'}
    Confidence: {response['emotion_result']['confidence'] if response['emotion_result'] else 0.0}
    Transcript: {response['transcript'] if response['transcript'] else 'N/A'}
    AI Response: {response['llm_response'] if response['llm_response'] else 'Please enter/analyze an audio file.'}""", 
    language='json'
    )

if __name__ == "__main__":
    main()