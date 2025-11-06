import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import load_ser_model
from src.inference import predict_emotion_from_file
from src.config import SUPPORTED_FORMATS, IDX_TO_EMOTION
from src.stt import load_stt_model, speech_to_text
from src.llm import create_llm_client, generate_response

ICON_PATH = "demo/assets/icon.png"
MFCC_IMAGE_PATH = "demo/assets/mfcc.png"
ZCR_IMAGE_PATH = "demo/assets/zcr.png"
RMS_IMAGE_PATH = "demo/assets/rms.png"

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon=ICON_PATH,
)

@st.cache_resource
def load_models():
    return {
        'ser': load_ser_model(),
        'stt': load_stt_model(), 
        'llm': create_llm_client()
    }

def demo():
    st.title("Speech Emotion Recognition + LLM")

    st.markdown(
        """
        Combines speech emotion recognition with large language models to create 
        emotionally intelligent responses. The actual 
        [notebook](https://www.kaggle.com/code/gemmin/speech-emotion-recognition-90) 
        is available for reference. Upload any audio file to analyze vocal emotions 
        and receive context-aware responses. Here are the supported emotions:
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
    response = {
        "emotion_result": {
            "emotion": "n/a",
            "confidence": "n/a",
            "all_scores": []
        },
        "transcript": "Your input audio transcript will appear here.",
        "llm_response": "The LLM-generated response will appear here."
    }

    if st.button("Analyze Audio", type="secondary", use_container_width=True) and uploaded_file is not None:
        response = get_response(uploaded_file, models)

    st.subheader("Response")

    st.json(response)

def feature_extraction():
    st.title("Feature Extraction")

    st.markdown(
        """
        We extract three features:
        1. Mel-Frequency Cepstral Coefficients (MFCC)
        2. Zero Crossing Rate (ZCR)
        3. Root Mean Square Energy (RMS)
        """
    )

    st.markdown(
        """
        **1. Mel-Frequency Cepstral Coefficients (MFCC)**
        They capture the spectral structure. The extraction process involves windowing the 
        signal, applying the Fourier transform, mapping the power spectrum onto 
        the perceptually-motivated mel scale, taking the logarithm to create 
        the mel spectrogram, & finally computing the discrete cosine transform 
        (DCT) to decorrelate the coefficients into the MFCC representation used 
        by the model. The visualization below shows both the mel spectrogram & 
        the MFCCs, but only the 13 MFCC coefficients are used as input features.
        """
    )
    st.latex(r"""
        \text{MFCC}_n = \sum_{k=1}^{K} \log(S_k) \cos\left[ n \left( k - \frac{1}{2} \right) \frac{\pi}{K} \right]
    """)
    st.image(MFCC_IMAGE_PATH, use_column_width=True)

    st.markdown(
        """
        **2. Zero-Crossing Rate (ZCR)**
        It measures how often the audio signal crosses zero amplitude. High ZCR
        means more noisy or hiss-like sounds; low ZCR means smoother, voiced sounds
        like vowels. This means ZCR can tell how "harsh" or "smooth," which is commonly
        associated with the emotion disgust.
        """
    )
    st.latex(r"""
        \text{ZCR} = \frac{1}{2N} \sum_{n=0}^{N-1} \left| \text{sgn}(x[n]) - \text{sgn}(x[n-1]) \right|
    """)
    st.image(ZCR_IMAGE_PATH)

    st.markdown(
        """
        **3. Root Mean Squared** 
        They measure the average energy or loudness. High RMS values indicate loud, 
        energetic speech segments, while low values correspond to quiet or whispered 
        speech. Thus, they help distinguish between high-energy emotions like anger 
        or happy (high RMS) vs. low-energy emotions like sadness or neutral (low RMS).
        """
    )
    st.latex(r"""
        \text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}
    """)
    st.image(RMS_IMAGE_PATH)

def model_info():
    st.title("Model Info")

    st.markdown(
        """
        The Speech Emotion Recognition model is a two-layer LSTM network designed
        in which the input shape is (352, 15), representing 352 time steps and 15 features
        (13 MFCC coefficients + ZCR + RMS). The total trainable parameters are around 54,000.  
        """
    )

    st.subheader("Model Architecture")
    st.code(
    '''
    def create_model():
        MODEL = Sequential()
        MODEL.add(layers.LSTM(64, return_sequences=True, input_shape=((352, 15))))
        MODEL.add(layers.LSTM(64))
        MODEL.add(layers.Dense(6, activation='softmax'))
            
        return MODEL
    '''
        , language="python"
    )

    hyperparams = {
        'Parameter': [
            'Epochs', 'Batch Size', 'Loss Function', 'Optimizer', 'Metrics', 'Class Weight'
        ],
        'Value': [
            '100', '6', 'categorical_crossentropy', 'RMSProp', 'categorical_accuracy', 'balanced'
        ]
    }
    
    df = pd.DataFrame(hyperparams)
    st.subheader("Model Hyperparameters")
    st.dataframe(df, use_container_width=True, hide_index=True)

def get_response(path, models):
    temp_path = f"temp_{path.name}"

    try: 
        with open(temp_path, "wb") as f:
            f.write(path.getbuffer())
                
        with st.spinner("Processing..."):
            emotion_result = predict_emotion_from_file(temp_path, model=models['ser'])
            emotion_result['all_scores'] = {
                IDX_TO_EMOTION[i]: round(float(score), 4) 
                for i, score in enumerate(emotion_result['all_scores'])
            }
            transcript = speech_to_text(temp_path, model=models['stt'])
            llm_response = generate_response(transcript, emotion_result['emotion'], models['llm'])

        return {
            'emotion_result': emotion_result,
            'transcript': transcript,
            'llm_response': llm_response.strip()
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    tab1, tab2, tab3 = st.tabs(["Demo", "Feature Extraction", "Model Info"])

    with tab1:
        demo()

    with tab2:
        feature_extraction()

    with tab3:
        model_info()