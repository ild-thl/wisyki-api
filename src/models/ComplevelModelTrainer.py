# Setup and imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.metrics import classification_report
from datetime import datetime
import time
import pickle
import json

class ComplevelModelTrainer():
    def __init__(self):
        pass

    def getReport(self):
        try:
            fileObject = open("./data/logs/report.json", "r")
        except FileNotFoundError:
            print("The file at ./data/logs/report.json" + " does not exist.")
            return None
        jsonContent = fileObject.read()
        return json.loads(jsonContent)
    

    def train(self, training_data):
        log = "Training results:"
        log += "\nDate: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # Start time.
        st = time.time()
        # Load data.
        labeled_data = pd.json_normalize(training_data)
        
        log += "\nTraining data size: " + str(labeled_data.shape[0])


        # Set input data.
        X = labeled_data.text

        # Set traget data.
        Y = labeled_data.label


        # Split data into test and training data.
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/2)


        # Create a model based on Multinominal Naive Bayes.
        model = make_pipeline(
            TfidfVectorizer(max_df=0.125, ngram_range=(1, 3), stop_words=stopwords.words('german')),
            OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None, alpha=0.001))
        )

        # Train the model with the train data.
        model.fit(X_train, y_train)

        # Create labels for the test data.
        prediction = model.predict(X_test)
        labels = ["A", "B", "C", "D"]
        log += "\n\n" + classification_report(y_test, prediction, target_names=labels, zero_division=0)
        report = classification_report(y_test, prediction, target_names=labels, zero_division=0, output_dict=True)

        pickle.dump(model, open("./data/models/comp-level_ai-model.pickle", 'wb'))

        # End time.
        et = time.time()

        # Get Elapsed time.
        elapsed_time = et - st
        log += "\n\nExecution time: " + str(elapsed_time) + ' seconds'

        report['time'] = st
        report['executiontime'] = elapsed_time
        report['modelname'] = 'naive-comp-level'

        jsonReport = json.dumps(report)
        jsonFile = open("./data/logs/report.json", "w")
        jsonFile.write(jsonReport)
        jsonFile.close()

        try:
            logFile = open("./data/logs/trainModelLog.txt", "a")
        except FileNotFoundError:
            print("The file at ./data/logs/trainModelLog.txt" + " does not exist.")

        logFile.write(log + "\n\n\n")
        logFile.close()

        return report


