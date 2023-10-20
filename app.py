from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from comp_level_predictor import comp_level_predictor
from topic_predictor import topic_predictor
from comp_level_model_trainer import comp_level_model_trainer
from skillfit_model_trainer import skillfit_model_trainer
from keyword_extractor import keyword_extractor
from esco_predictor import esco_predictor
from vectorsearcher import vectorsearcher
from chatsearcher import chatsearcher
from semanticsorter import semanticsorter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.vectorstores import Chroma
import os
import openai
import pdfplumber
import json
from recog_ai import recognition_assistant
from dotenv import load_dotenv

project_folder = os.path.expanduser('~/wisyki-api')
load_dotenv(os.path.join(project_folder, '.env'))


app = Flask(__name__)
CORS(app)


@app.before_first_request
def load_instructor():
    global instructor
    instructor = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        embed_instruction="Represent the document for retrieval: ",
        query_instruction="Represent the query for retrieval: "
    )
    dir = os.path.dirname(__file__)
    persist_directory = dir + '/data/esco_vectorstore'
    chroma_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    global vectordb
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=instructor, client_settings=chroma_settings)

    persist_directory = dir + '/data/thl_modules_vectorstore'
    chroma_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    
    global moduledb
    moduledb = Chroma(persist_directory=persist_directory, embedding_function=instructor, client_settings=chroma_settings)
    
    global recognition_ai
    recognition_ai = recognition_assistant(moduledb)
        


def load_moduledb(embedding):
    return Chroma(
        client=chromadb.PersistentClient(
            os.path.dirname(__file__) + "/data/thl_modules_vectorstore"
        ),
        embedding_function=embedding,
        client_settings=Settings(anonymized_telemetry=False),
    )


embedding = load_embedding()
skilldb = load_skilldb(embedding)
moduledb = load_moduledb(embedding)

        if not doc:
            return render_template('module_suggestions.html')

        # No more than 10000 characters
        doc = doc[:10000]
        
        module_suggestions = recognition_ai.getModuleSuggestions(doc)
        external_module_json = recognition_ai.getModulInfo(doc)
        external_module_parsed = json.loads(external_module_json)

        return render_template('module_suggestions.html', module_suggestions=module_suggestions, external_module_parsed=external_module_parsed, external_module_json=external_module_json)
        
    return render_template('module_suggestions.html')

# Endpunkt für die Startseite
@app.route("/find_module", methods=["GET", "POST"])
def find_module():
    if request.method == "POST":
        # Hier verarbeiten wir den Dateiupload und rufen getModuleSuggestions() auf.
        doc = None
        uploaded_file = request.files["file"]
        if uploaded_file:
            # Check if it's a PDF file
            if uploaded_file.filename.endswith(".pdf"):
                with pdfplumber.open(uploaded_file) as pdf:
                    doc = ""
                    for page_num in range(max(2, len(pdf.pages))):
                        doc += pdf.pages[page_num].extract_text()
            # Check if it's a TXT file
            elif uploaded_file.filename.endswith(".txt"):
                doc = uploaded_file.read().decode("utf-8")
            # Check if it's a xml file
            elif uploaded_file.filename.endswith(".xml"):
                doc = uploaded_file.read().decode("utf-8")
            else:
                raise Exception("File type not supported")
        else:
            doc = request.form["text"]

        if not doc:
            return render_template("module_suggestions.html")

        # No more than 10000 characters
        doc = doc[:10000]

        recog_assistant = recognition_assistant(moduledb)
        module_suggestions = recog_assistant.getModuleSuggestions(doc)
        external_module_json = recog_assistant.getModulInfo(doc)
        external_module_parsed = json.loads(external_module_json)

        return render_template(
            "module_suggestions.html",
            module_suggestions=module_suggestions,
            external_module_parsed=external_module_parsed,
            external_module_json=external_module_json,
        )

    return render_template("module_suggestions.html")


# Endpunkt für die Modulauswahl und Prüfung
@app.route("/select_module", methods=["POST"])
def select_module():
    recog_assistant = recognition_assistant(moduledb)
    internal_module_json = request.form["selected_module"]
    internal_module_parsed = json.loads(internal_module_json)

    # Get learninggoals
    internal_module_ai_json = recog_assistant.getModulInfo(internal_module_json)
    internal_module_ai_parsed = json.loads(internal_module_ai_json)
    internal_module_parsed["learninggoals"] = internal_module_ai_parsed["learninggoals"]

    external_module_json = request.form["external_module"]
    external_module_parsed = json.loads(external_module_json)

    # Hier rufen wir getExaminationResult() auf und generieren das Prüfungsergebnis.
    examination_result = recog_assistant.getExaminationResult(
        internal_module_json, external_module_json
    )

    return render_template(
        "examination_result.html",
        internal_module_parsed=internal_module_parsed,
        external_module_parsed=external_module_parsed,
        examination_result=examination_result,
    )


@app.route("/predictCompLevel", methods=["POST"])
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

@app.route("/trainSkillfit", methods=['GET'])
def train_skillfit():
    trainer = skillfit_model_trainer()
    training_stats = trainer.train()
    return jsonify(training_stats)


@app.route("/getSkillfitReport", methods=['GET'])
def report_skillfit():
    trainer = skillfit_model_trainer()
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

    escosearcher = vectorsearcher(vectordb, instructor)
    skills = escosearcher.predict(doc, 20, 0, .2, [], [])

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
    
    filterconcepts = []
    if 'filterconcepts' in data:
        filterconcepts = data["filterconcepts"]

    strict = 0
    if 'strict' in data:
        strict = int(data["strict"])

    skills = []
    if 'skills' in data:
        skills = data["skills"]

    trusted_score = .2
    if 'trusted_score' in data:
        trusted_score = float(data["trusted_score"])

    searchervector = vectorsearcher(vectordb, instructor)
    skills = searchervector.predict(doc, top_k, strict, trusted_score, skills, filterconcepts)

    return jsonify(skills)


@app.route("/chatsearch", methods=['POST'])
def chatsearch():
    data = request.get_json()

    doc = None
    if 'doc' in data:
        doc = data["doc"]

    los = []
    if 'los' in data:
        los = data["los"]

    skills = []
    if 'skills' in data:
        skills = data["skills"]
    
    filterconcepts = []
    if 'filterconcepts' in data:
        filterconcepts = data["filterconcepts"]

    top_k = 20
    if 'top_k' in data:
        top_k = int(data["top_k"])

    strict = 0
    if 'strict' in data:
        strict = int(data["strict"])

    trusted_score = .2
    if 'trusted_score' in data:
        trusted_score = float(data["trusted_score"])

    temperature = .05
    if 'temperature' in data:
        temperature = float(data["temperature"])

    openai_api_key = ""
    if 'openai_api_key' in data:
        openai_api_key = data["openai_api_key"]

    request_timeout = 20
    if 'request_timeout' in data:
        request_timeout = int(data["request_timeout"])

    llm_validation = False
    if 'llm_validation' in data:
        llm_validation = bool(data["llm_validation"])

    skillfit_validation = False
    if 'skillfit_validation' in data:
        skillfit_validation = bool(data["skillfit_validation"])

    searcherchat = chatsearcher(vectordb, instructor)
    
    try:
        skills = searcherchat.predict(doc, los, skills, filterconcepts, top_k, strict, trusted_score, temperature, openai_api_key, request_timeout, llm_validation, skillfit_validation)
        return jsonify(skills), 200
    except openai.error.Timeout:
        # Catch timeout error and send 502 response.
        return jsonify({
                'status': 408,
                'message': 'OpenAI API Timeout'
            }), 408


@app.route("/semanticsort", methods=['POST'])
def semanticsort():
    data = request.get_json()
    base = ''
    if 'base' in data and data["base"] != '':
        base = data["base"]
    else:
        return jsonify({
                'status': 400,
                'message': 'Missing or empty base value.'
            }), 400

    documents = []
    if 'documents' in data and len(data["documents"]) > 0:
        documents = data["documents"]
    else:
        return jsonify({
                'status': 400,
                'message': 'Missing or empty documents.'
            }), 400
    
    sorter = semanticsorter(instructor)
    return jsonify(sorter.sort(base, documents)), 200

@app.route("/getEmbeddings", methods=['POST'])
def get_embeddings():
    data = request.get_json()
    documents = ''
    if 'docs' in data and len(data["docs"]):
        documents = data["docs"]
    else:
        return jsonify({
                'status': 400,
                'message': 'Missing or empty docs value.'
            }), 400
    
    return jsonify(instructor.embed_documents(documents)), 200

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

    escopredictor = esco_predictor(instructor)
    skills = escopredictor.predict(searchterms, extract_keywords,
                               schemes, filterconcepts, min_relevancy, exclude_irrelevant, doc)

    return jsonify(skills)


if __name__ == '__main__':
    app.run(debug=True)
