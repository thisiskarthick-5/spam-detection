from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model, vectorizer = pickle.load(open("spam_model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data['message']

    # Transform input
    transformed = vectorizer.transform([message])

    prediction = model.predict(transformed)[0]

    result = "Spam" if prediction == 1 else "Not Spam"

    return jsonify({
        "prediction": result
    })

if __name__ == '__main__':
    app.run(debug=True)