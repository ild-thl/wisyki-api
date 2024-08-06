# Setup and imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
from sklearn.metrics import classification_report
from datetime import datetime
import time
import pickle
import json
import os

## 1000 p.C Evaluation
## Train:
# 300 p.C = .58 acc
# 600 p.C = .62 acc
# 900 p.C = .63 acc
# 1500 p.C = .68 acc
# 3000 p.C = .61 acc
# 6000 p.C = .68 acc
# 9000 p.C = .68 acc
# 12000 p.C = .75 acc


# 12000 p.C and val = eval = .64 acc
# 12000 p.C w. val and val = eval = .87 acc


class ComplevelModelTrainer:
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

    def train(self):
        log = "Training results:"
        log += "\nDate: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # Start time.
        st = time.time()
        # Load training data
        train_data = pd.read_json(
            "./data/models/comp_level_model/synthetic-comp-levels.json",
            orient="records",
        )
        val_data = pd.read_json(
            "./data/models/comp_level_model/validated-comp-levels.json",
            orient="records",
        )
        # add val_data to train_data
        # Get class with the least number of samples
        min_samples = train_data["label"].value_counts().min()
        print("Min samples: ", min_samples)
        eval_samples = round(min_samples/5)
        train_samples = min_samples - eval_samples
        # Get min_samples samples from each class
        sampled_data = train_data.groupby("label").head(train_samples + eval_samples)
        train_data = sampled_data.groupby("label").head(train_samples)
        train_data = pd.concat([train_data, val_data], ignore_index=True)
        eval_data = sampled_data.groupby("label").tail(eval_samples)
        eval_data = pd.concat([eval_data, val_data], ignore_index=True)
        
        log += "\nTraining data size: " + str(train_data.shape[0])

        # X = train_data["text"]
        # Y = train_data["label"]
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1/5)
        X_train = train_data["text"]
        y_train = train_data["label"]
        X_test = eval_data["text"]
        y_test = eval_data["label"]

        # Create a model based on Multinominal Naive Bayes.
        model = make_pipeline(
            TfidfVectorizer(
                max_df=0.125, ngram_range=(1, 3), stop_words=stopwords.words("german")
            ),
            OneVsRestClassifier(
                MultinomialNB(fit_prior=True, class_prior=None, alpha=0.001)
            ),
        )

        # Train the model with the train data.
        model.fit(X_train, y_train)

        # Create labels for the test data.
        prediction = model.predict(X_test)
        labels = ["A", "B", "C"]
        log += "\n\n" + classification_report(
            y_test, prediction, target_names=labels, zero_division=0
        )
        report = classification_report(
            y_test, prediction, target_names=labels, zero_division=0, output_dict=True
        )

        pickle.dump(
            model,
            open("./data/models/comp_level_model/comp-level_ai-model.pickle", "wb"),
        )

        # End time.
        et = time.time()

        # Get Elapsed time.
        elapsed_time = et - st
        log += "\n\nExecution time: " + str(elapsed_time) + " seconds"

        report["time"] = st
        report["executiontime"] = elapsed_time
        report["modelname"] = "naive-comp-level"

        jsonReport = json.dumps(report)

        # Ensure the directory exists
        logs_dir = "./data/logs/"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Now, safely write the report.json file
        jsonFile = open(os.path.join(logs_dir, "report.json"), "w")
        jsonFile.write(jsonReport)
        jsonFile.close()

        try:
            logFile = open("./data/logs/trainModelLog.txt", "a")
        except FileNotFoundError:
            print("The file at ./data/logs/trainModelLog.txt" + " does not exist.")

        logFile.write(log + "\n\n\n")
        logFile.close()

        return report
