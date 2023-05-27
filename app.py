from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from comp_level_predictor import comp_level_predictor
from topic_predictor import topic_predictor
from comp_level_model_trainer import comp_level_model_trainer
from keyword_extractor import keyword_extractor
from esco_predictor import esco_predictor
from vectorsearcher import vectorsearcher


app = Flask(__name__)
CORS(app)


@app.before_first_request
def load_instructor():
    global instructor
    instructor = esco_predictor()


@app.before_first_request
def load_escosearcher():
    global escosearcher
    escosearcher = vectorsearcher()


@app.route("/", methods=['GET'])
def index():
    return jsonify({"about": "This is an API providing AI-predictions for WISY@KI"})


@app.route("/predictCompLevel", methods=['POST'])
def predict_complevel():
    data = request.get_json()
    title = data["title"]
    description = data["description"]
    complevelmodel = comp_level_predictor()
    prediction = complevelmodel.predict(title, description)

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
    seed = data["title"]
    text = data["text"]
    bertmodel = keyword_extractor()
    keywords = bertmodel.extract_keywords(text, seed)

    return jsonify(keywords)


@app.route('/escoAutomat', methods=['GET'])
def home():
    return render_template('predict_esco_home.html')


@app.route('/predictESCOWeb', methods=['POST'])
def predictESCOWeb():
    doc = request.form['input_text']

    skills = escosearcher.predict(doc, 20)

    return render_template('predict_esco_home.html', result=skills['results'])


@app.route("/vectorsearch", methods=['POST'])
def vectorsearch():
    data = request.get_json()

    doc = None
    if 'doc' in data:
        doc = data["doc"]

    top_k = 20
    if 'top_k' in data:
        top_k = int(data["top_k"])

    skills = escosearcher.predict(doc, top_k)

    return jsonify(skills)


@app.route("/predictESCO", methods=['POST'])
def predict_skills():
    data = request.get_json()

    searchterms = {}
    if 'searchterms' in data:
        searchterms = data["searchterms"]

    doc = None
    if 'doc' in data:
        doc = data["doc"]

    min_relevancy = None
    if 'min_relevancy' in data:
        min_relevancy = float(data["min_relevancy"])

    exclude_irrelevant = True
    if 'exclude_irrelevant' in data:
        exclude_irrelevant = bool(data["exclude_irrelevant"])

    extract_keywords = False
    if 'extract_keywords' in data:
        extract_keywords = bool(data["extract_keywords"])

    filterconcepts = []
    if 'filterconcepts' in data:
        filterconcepts = data["filterconcepts"]

    schemes = "http://data.europa.eu/esco/concept-scheme/member-skills, http://data.europa.eu/esco/concept-scheme/skills-hierarchy"
    if 'schemes' in data:
        schemes = data["schemes"]

    skills = instructor.predict(searchterms, extract_keywords,
                               schemes, filterconcepts, min_relevancy, exclude_irrelevant, doc)

    return jsonify(skills)


if __name__ == '__main__':
    app.run(debug=True)
