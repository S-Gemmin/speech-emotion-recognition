from tensorflow.keras.models import Sequential
from tensorflow.keras import layers 
from src.config import INPUT_SHAPE, LSTM_UNITS_1, LSTM_UNITS_2, NUM_EMOTIONS, ACTIVATION, LOSS, OPTIMIZER, MODEL_PATH

def load_ser_model(): 
    try:
        model = Sequential()
        model.add(layers.LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=INPUT_SHAPE))
        model.add(layers.LSTM(LSTM_UNITS_2))
        model.add(layers.Dense(NUM_EMOTIONS, activation=ACTIVATION))
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=['accuracy'])
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        print(e)
        return None