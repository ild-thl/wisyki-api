import joblib
import os
import json


class topic_predictor:
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        self.model = self.deserialize()
        self.labels = json.load(open(self.dir + "/models/topic_model_labels.json"))
        pass

    def deserialize(self):
        with open(self.dir + "/models/topic_model.pickle", "rb") as handle:
            model = joblib.load(handle)
            return model

    def predict(self, doc):
        probabilities = self.model.predict_proba([doc]).tolist()[0]
        # Sort probabilities in descending order and myp to topics.
        topics = sorted(list(zip(self.labels, probabilities)), key=lambda x: x[1], reverse=True)

        print(topics)

        return topics
