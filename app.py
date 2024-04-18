import joblib
import re
from flask import Flask, request, jsonify, render_template
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Flask app
app = Flask(__name__)

# Initialize Firestore with credentials
cred = credentials.Certificate("/Users/shreyastiwary/openai/serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


@app.route('/')
def landing():
    return render_template('landing.html')
@app.route('/index')
def index():
    return render_template('index.html')
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('domain_model.h5')
tokenizer = Tokenizer()
with open('label_encoder.joblib', 'rb') as f:
    label_encoder = joblib.load(f)

# Preprocess text data
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    return text

# Function to predict labels for input complaint text
def predict_label(complaint_text):
    preprocessed_text = preprocess_text(complaint_text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    predicted_label_index = np.argmax(model.predict(padded_sequence), axis=-1)[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
    return predicted_label

@app.route('/cate', methods=['POST'])
def predict():
    data = request.get_json()
    complaint_text = data['complaint']
    predicted_label = predict_label(complaint_text)
    return jsonify({'predicted_domain': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
