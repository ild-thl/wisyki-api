# Setup and imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
import joblib
import os
import json
from datetime import datetime
import time


class topic_model_trainer:
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        pass

    def getReport(self):
        if os.path.exists(self.dir + "/logs/trainTopicModelReport.json"):
            fileObject = open(self.dir + "/logs/trainTopicModelReport.json", "r")
            jsonContent = fileObject.read()
            return json.loads(jsonContent)
        else:
            return None

    def train(self):
        log = "Training results:"
        log += "\nDate: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # Start time.
        st = time.time()

        # Load data.
        df = pd.read_json(
            self.dir + "/data/preparation/topic_dataset.json", orient="records"
        )

        report = {}
        report["dataset_size"] = df.shape[0]
        log += "\nTraining data size: " + str(df.shape[0])

        # Set input data.
        X = df.text

        # Set traget data.
        Y = df.thema

        # Split data into test and training data.
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3)

        # Create a model based on Multinominal Naive Bayes.
        model = make_pipeline(
            TfidfVectorizer(
                max_df=0.5, ngram_range=(1, 2), stop_words=stopwords.words("german")
            ),
            OneVsRestClassifier(
                MultinomialNB(fit_prior=True, class_prior=None, alpha=0.001)
            ),
        )

        # Train the model with the train data.
        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, self.dir + "/models/topic_model.pickle", compress=3)

        # Create labels for the test data.
        prediction = model.predict(X_test)

        labels = list(set(y_test))
        # Save models to file
        json.dump(labels, open(self.dir + "/models/topic_model_labels.json", "w"))

        report["labels"] = labels

        # Create confusion matrix and heat map.
        report["evaluation"] = classification_report(
            y_test,
            prediction,
            target_names=list(labels),
            zero_division=0,
            output_dict=True,
        )

        log += str(report)

        # End time.
        et = time.time()

        # Get Elapsed time.
        elapsed_time = et - st
        log += "\n\nExecution time: " + str(elapsed_time) + " seconds"
        print(log)

        report["time"] = st
        report["executiontime"] = elapsed_time
        report["modelname"] = "topic"

        with open(self.dir + "/logs/trainTopicModelReport.json", "w") as jsonFile:
            json.dump(report, jsonFile)

        # Check if "/logs/trainSkillfitModelLog.txt exists.
        if os.path.exists(self.dir + "/logs/trainTopicModelLog.txt"):
            logFile = open(self.dir + "/logs/trainTopicModelLog.txt", "a")
        else:
            logFile = open(self.dir + "/logs/trainTopicModelLog.txt", "w")

        logFile.write(log + "\n\n\n")
        logFile.close()

        return report
