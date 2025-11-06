import numpy as np
import librosa
from pydub import AudioSegment

from src.config import FRAME_LENGTH, HOP_LENGTH, MAX_AUDIO_LENGTH, N_MFCC, SAMPLE_RATE, TOP_DB

def preprocess_audio(path): # 22050 hz
    raw_audio = AudioSegment.from_file(path)
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=TOP_DB)

    if len(trimmed) > MAX_AUDIO_LENGTH:
        return trimmed[:MAX_AUDIO_LENGTH]
    else:
        return np.pad(trimmed, (0, MAX_AUDIO_LENGTH-len(trimmed)), 'constant')

def extract_features(y):
    zcr_list = []
    rms_list = []
    mfccs_list = []

    try:
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        mfccs = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
        
        zcr_list.append(zcr)
        rms_list.append(rms)
        mfccs_list.append(mfccs)
    except Exception as e:
        print(e)
        return None
    
    X = np.concatenate((
        np.swapaxes(zcr_list, 1, 2), 
        np.swapaxes(rms_list, 1, 2), 
        np.swapaxes(mfccs_list, 1, 2)), 
        axis=2
    )
    X = X.astype('float32')

    return X