from vader_library.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.get("/")
def heartbeat():
    return "I'm alive!"

@app.get("/sentimentIntensity/")
def sentimentIntensity():
    vader = SentimentIntensityAnalyzer()

    sentimentDictionary = jsonify(vader.polarity_scores(str(request.args)))

    return sentimentDictionary