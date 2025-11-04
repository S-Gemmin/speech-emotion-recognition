# speech-emotion-recognition


speech-emotion-recognition/
├── README.md                 
├── requirements.txt         
├── setup.py                  
├── .gitignore               
├── LICENSE
│
├── speech-emotion-recognition-90.ipynb  # Original Kaggle notebook              
│
├── src/
│   └── __init__.py
│   └── model.py         # Model architecture & loading
│   └── preprocessing.py # Audio feature extraction
│   └── inference.py     # Prediction pipeline
│   └── config.py        # Hyperparameters & paths
│   └── utils.py         # Helper functions
│
├── demo/
│   ├── streamlit_app.py     # Streamlit demo
│   ├── requirements.txt     # Demo-specific deps
│   └── assets/              # UI assets
│
├── models/
│   └── README.md            # Model download instructions
│