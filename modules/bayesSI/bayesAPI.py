import nltk
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.get("/")
def heartbeat():
    return "I'm alive!"

@app.get("/sentimentIntensity/")
def sentimentIntensity():

    return 1

if __name__ == "__main__":
    app.run(debug=True)