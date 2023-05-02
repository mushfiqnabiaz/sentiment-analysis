import os
import json
import pickle
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify

# Load the Keras model and tokenizer
model_file = './model/model.json'
weights_file = './model/model.h5'
tokenizer_file = './model/tokenizer.pickle'

with open(model_file, 'r') as f:
    model_json = f.read()
    model = model_from_json(model_json)

model.load_weights(weights_file)

with open(tokenizer_file, 'rb') as f:
    tokenizer = pickle.load(f)

# Set up the Flask app
app = Flask(__name__)

# Define the endpoint for sentiment prediction


@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Get the text data from the request
    text_data = request.get_data(as_text=True)

    # Tokenize the text data
    tokenized_data = tokenizer.texts_to_sequences([text_data])

    # Pad the tokenized data to a fixed length
    max_len = 48
    padded_data = pad_sequences(tokenized_data, maxlen=max_len)

    # Perform sentiment prediction using the model
    prediction = model.predict(padded_data)[0]
    tag = np.argmax(prediction)
    tag_confidence = prediction[tag]
    sentiment_confidence = np.max(prediction)

    # Return the results in JSON format
    results = {
        'tag': tag,
        'tag_confidence': float(tag_confidence),
        'sentiment_confidence': float(sentiment_confidence)
    }

    return jsonify(str(results))


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
