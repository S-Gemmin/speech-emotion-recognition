import whisper

from src.config import STT_TYPE

def load_stt_model():
    try: 
        stt_model = whisper.load_model(STT_TYPE)
        return stt_model
    except Exception as e:
        print(e)
        return None
    
def speech_to_text(path, model):
    try:
        result = model.transcribe(path)
        return result['text']
    except Exception as e:
        print(e)
        return None
