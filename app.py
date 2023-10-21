from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from chromadb.config import Settings
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import pdfplumber
import json
import os
from dotenv import load_dotenv
from comp_level_predictor import comp_level_predictor
from topic_predictor import topic_predictor
from topic_model_trainer import topic_model_trainer
from comp_level_model_trainer import comp_level_model_trainer
from skillfit_model_trainer import skillfit_model_trainer
from skillfit_predictor import skillfit_predictor
from keyword_extractor import keyword_extractor
from esco_predictor import esco_predictor
from vectorsearcher import vectorsearcher
from chatsearcher import chatsearcher
from recog_ai import recognition_assistant


project_folder = os.path.expanduser("~/wisykiapi")
load_dotenv(os.path.join(project_folder, ".env"))

app = Flask(__name__)
CORS(app)


def load_embedding():
    return HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        embed_instruction="Represent the document for retrieval: ",
        query_instruction="Represent the query for retrieval: ",
    )


def load_escodb(embedding):
    return Chroma(
        client=chromadb.PersistentClient(
            os.path.dirname(__file__) + "/data/esco_vectorstore"
        ),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


def load_dkzdb(embedding):
    return Chroma(
        client=chromadb.PersistentClient(
            os.path.dirname(__file__) + "/data/dkz_competencies_vectorstore"
        ),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


def load_moduledb(embedding):
    return Chroma(
        client=chromadb.PersistentClient(
            os.path.dirname(__file__) + "/data/thl_modules_vectorstore"
        ),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


def load_skillfit_model():
    return skillfit_predictor()


def load_topic_model():
    return topic_predictor()


embedding = load_embedding()
escodb = load_escodb(embedding)
dkzdb = load_dkzdb(embedding)
moduledb = load_moduledb(embedding)
skillfit_model = load_skillfit_model()
topic_model = load_topic_model()


@app.route("/", methods=["GET"])
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


@app.route("/trainTopicModel", methods=["GET"])
def train_topic():
    trainer = topic_model_trainer()
    training_stats = trainer.train()
    return jsonify(training_stats)


@app.route("/getTopicModelReport", methods=["GET"])
def report_topic():
    trainer = topic_model_trainer()
    report = trainer.getReport()
    return jsonify(report)


@app.route("/predictTopic", methods=["POST"])
def predict_topic():
    data = request.get_json()
    doc = data["doc"]
    prediction = topic_model.predict(doc)

    return jsonify(prediction)


@app.route("/trainCompLevel", methods=["POST"])
def train_complevel():
    data = request.get_json()
    trainer = comp_level_model_trainer()
    training_stats = trainer.train(data)
    return jsonify(training_stats)


@app.route("/getCompLevelReport", methods=["GET"])
def report_complevel():
    trainer = comp_level_model_trainer()
    report = trainer.getReport()
    return jsonify(report)


@app.route("/trainSkillfit", methods=["GET"])
def train_skillfit():
    trainer = skillfit_model_trainer()
    training_stats = trainer.train()
    return jsonify(training_stats)


@app.route("/getSkillfitReport", methods=["GET"])
def report_skillfit():
    trainer = skillfit_model_trainer()
    report = trainer.getReport()
    return jsonify(report)


@app.route("/extractKeywords", methods=["POST"])
def extract_keywords():
    data = request.get_json()
    seed = data["title"]
    text = data["text"]
    bertmodel = keyword_extractor()
    keywords = bertmodel.extract_keywords(text, seed)

    return jsonify(keywords)


@app.route("/escoAutomat", methods=["GET"])
def home():
    return render_template("predict_esco_home.html")


@app.route("/predictESCOWeb", methods=["POST"])
def predictESCOWeb():
    doc = request.form["input_text"]

    escosearcher = vectorsearcher(vectordb, instructor)
    skills = escosearcher.predict(doc, 20, 0, 0.2, [], [])

    return render_template("predict_esco_home.html", result=skills["results"])


@app.route("/vectorsearch", methods=["POST"])
def vectorsearch():
    data = request.get_json()

    doc = None
    if "doc" in data:
        doc = data["doc"]

    top_k = 20
    if "top_k" in data:
        top_k = int(data["top_k"])

    filterconcepts = []
    if "filterconcepts" in data:
        filterconcepts = data["filterconcepts"]

    strict = 0
    if "strict" in data:
        strict = int(data["strict"])

    skills = []
    if "skills" in data:
        skills = data["skills"]

    trusted_score = 0.2
    if "trusted_score" in data:
        trusted_score = float(data["trusted_score"])

    searchervector = vectorsearcher(escodb, embedding)
    skills = searchervector.predict(
        doc, top_k, strict, trusted_score, skills, filterconcepts
    )

    return jsonify(skills)


@app.route("/chatsearch", methods=["POST"])
def chatsearch():
    data = request.get_json()

    doc = None
    if "doc" in data:
        doc = data["doc"]

    los = []
    if "los" in data:
        los = data["los"]

    skills = []
    if "skills" in data:
        skills = data["skills"]

    filterconcepts = []
    if "filterconcepts" in data:
        filterconcepts = data["filterconcepts"]

    top_k = 20
    if "top_k" in data:
        top_k = int(data["top_k"])

    strict = 0
    if "strict" in data:
        strict = int(data["strict"])

    trusted_score = 0.2
    if "trusted_score" in data:
        trusted_score = float(data["trusted_score"])

    temperature = 0.05
    if "temperature" in data:
        temperature = float(data["temperature"])

    use_llm = False
    if "use_llm" in data:
        use_llm = bool(data["use_llm"])

    request_timeout = 20
    if "request_timeout" in data:
        request_timeout = int(data["request_timeout"])

    llm_validation = False
    if "llm_validation" in data:
        llm_validation = bool(data["llm_validation"])

    skillfit_validation = False
    if "skillfit_validation" in data:
        skillfit_validation = bool(data["skillfit_validation"])

    skilldb = escodb
    skill_taxonomy = "ESCO"
    if "skill_taxonomy" in data:
        skill_taxonomy = data["skill_taxonomy"]
        if skill_taxonomy == "ESCO":
            skilldb = escodb
        elif skill_taxonomy == "DKZ":
            skilldb = dkzdb
        else:
            return (
                jsonify({"status": 400, "message": "Invalid skill_taxonomy value."}),
                400,
            )

    searcherchat = chatsearcher(embedding, skillfit_model)

    skills = searcherchat.predict(
        skilldb,
        skill_taxonomy,
        doc,
        los,
        skills,
        filterconcepts,
        top_k,
        strict,
        trusted_score,
        temperature,
        use_llm,
        request_timeout,
        llm_validation,
        skillfit_validation,
    )
    return jsonify(skills), 200


@app.route("/getEmbeddings", methods=["POST"])
def get_embeddings():
    data = request.get_json()
    documents = ""
    if "docs" in data and len(data["docs"]):
        documents = data["docs"]
    else:
        return jsonify({"status": 400, "message": "Missing or empty docs value."}), 400

    return jsonify(embedding.embed_documents(documents)), 200


@app.route("/predictESCO", methods=["POST"])
def predict_skills():
    data = request.get_json()

    searchterms = {}
    if "searchterms" in data:
        searchterms = data["searchterms"]

    doc = None
    if "doc" in data:
        doc = data["doc"]

    min_relevancy = None
    if "min_relevancy" in data:
        min_relevancy = float(data["min_relevancy"])

    exclude_irrelevant = True
    if "exclude_irrelevant" in data:
        exclude_irrelevant = bool(data["exclude_irrelevant"])

    extract_keywords = False
    if "extract_keywords" in data:
        extract_keywords = bool(data["extract_keywords"])

    filterconcepts = []
    if "filterconcepts" in data:
        filterconcepts = data["filterconcepts"]

    schemes = "http://data.europa.eu/esco/concept-scheme/member-skills, http://data.europa.eu/esco/concept-scheme/skills-hierarchy"
    if "schemes" in data:
        schemes = data["schemes"]

    escopredictor = esco_predictor(embedding)
    skills = escopredictor.predict(
        searchterms,
        extract_keywords,
        schemes,
        filterconcepts,
        min_relevancy,
        exclude_irrelevant,
        doc,
    )

    return jsonify(skills)


if __name__ == "__main__":
    app.run(debug=True)
