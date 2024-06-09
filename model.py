from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import random
import json

# Initialize the stemmer
stemmer = PorterStemmer()
# Load predefined responses and tokens
with open('intents.json', 'r') as file:
    intents = json.load(file)


def get_response(user_input):
    tokens = word_tokenize(user_input.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    for intent, data in intents.items():
        stemmed_intent_tokens = [stemmer.stem(token) for token in data["tokens"]]
        if any(token in stemmed_tokens for token in stemmed_intent_tokens):
            return random.choice(data["responses"])

    return random.choice(intents["default"]["responses"])


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/get_response', methods=['POST'])
def get_chatbot_response():
    user_input = request.form["user_input"]
    response = get_response(user_input)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
