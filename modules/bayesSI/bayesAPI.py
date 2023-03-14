from components.bayes import classify, accuracy
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.get("/")
def heartbeat():
    return "I'm alive!"

@app.get("/datasetAccuracy/")
def datasetAccuracy():
    return jsonify(round(accuracy(), 2))

@app.get("/sentimentPolarity/")
def sentimentIntensity():
    userInput = request.args.to_dict()['']
    return jsonify(classify(userInput))

if __name__ == "__main__":
    app.run(debug=True)