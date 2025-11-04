import numpy as np
from src.config import IDX_TO_EMOTION
from src.preprocessing import extract_features, preprocess_audio

def predict_emotion_from_file(path, model):
    y = preprocess_audio(path)
    if y is None:
        print("Audio preprocessing failed.")
        return None
    
    features = extract_features(y)
    if features is None:
        print("No features extracted.")
        return None
    
    try:
        predictions = model.predict(features)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_emotion = IDX_TO_EMOTION[predicted_index]
        confidence = predictions[0][predicted_index]

        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'all_scores': predictions[0].tolist()
        }
    except Exception as e:
        print(e)
        return None