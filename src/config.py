"""
Configuration file for Speech Emotion Recognition
Contains all constants, paths, and hyperparameters
"""

import os

SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13
TOP_DB = 25  # Trimming threshold
MAX_AUDIO_LENGTH = 180000  # Maximum audio samples (approx 8 seconds at 22050 Hz)

# SER Hyperparameters
INPUT_SHAPE = (352, 15)  # (time_steps, features)
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 64
NUM_EMOTIONS = 6
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'
OPTIMIZER = 'RMSProp'

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust']
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

# STT 
STT_TYPE = 'base.en'

# LLM
LLM_TYPE = 'llama-3.1-8b-instant'
MAX_TOKENS = 100
TEMPERATURE = 0.7

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'ser.h5')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

SUPPORTED_FORMATS = ['.wav', '.mp3']

MODEL_INFO = {
    'name': 'Speech Emotion Recognition LSTM',
    'version': '1.0.0',
    'accuracy': '80%+', # since combined male & female
    'datasets': ['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS'],
    'architecture': 'LSTM',
    'input_features': 'ZCR + RMS + MFCC (13 coefficients)'
}
