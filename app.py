from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
model, vectorizer = pickle.load(open("spam_model.pkl", "rb"))

@app.route('/')
def home():
    return "Spam API is running"

@app.route('https://spam-detection-ajtk.onrender.com/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']

    transformed = vectorizer.transform([message])
    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({"prediction": result})