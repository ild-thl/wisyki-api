from flask import Flask,jsonify,request
from flask_cors import CORS
from comp_level_predictor import comp_level_predictor
from topic_predictor import topic_predictor
from comp_level_model_trainer import comp_level_model_trainer
from keyword_extractor import keyword_extractor

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def index():
    return jsonify({"about": "This is an API providing AI-predictions for WISY@KI"})


@app.route("/predictCompLevel", methods=['POST'])
def predict_complevel():
    data = request.get_json()
    title = data["title"]
    description = data["description"]
    model = comp_level_predictor()
    prediction = model.predict(title, description)
    
    return jsonify(prediction)


@app.route("/predictTopic", methods=['POST'])
def predict_topic():
    data = request.get_json()
    title = data["title"]
    description = data["description"]
    model = topic_predictor()
    prediction = model.predict(title, description)
    
    return jsonify(prediction)

@app.route("/trainCompLevel", methods=['POST'])
def train_complevel():
    data = request.get_json()
    trainer = comp_level_model_trainer()
    training_stats = trainer.train(data)
    return jsonify(training_stats)

@app.route("/getCompLevelReport", methods=['GET'])
def report_complevel():
    trainer = comp_level_model_trainer()
    report = trainer.getReport()
    return jsonify(report)


@app.route("/extractKeywords", methods=['POST'])
def extract_keywords():
    data = request.get_json()
    text = data["text"]
    model = keyword_extractor()
    keywords = model.extract_keywords(text)
    
    return jsonify(keywords)


if __name__ == '__main__':
    app.run(debug=True)