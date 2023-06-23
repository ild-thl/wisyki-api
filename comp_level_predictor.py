import pickle
import os

class comp_level_predictor():
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        pass
  
    def deserialize(self):
        with open(self.dir + '/data/comp-level_ai-model.pickle', 'rb') as handle:
            model = pickle.load(handle)
            return model
  
    def predict(self, title, description):
        text = title + " \n\n " + description
        model = self.deserialize()
        prediction = model.predict([text]).tolist()[0]
        probability = model.predict_proba([text]).tolist()[0]
        labels = ['A', 'B', 'C']
        level = prediction[0]
        prediction_proba = probability[labels.index(level)]
        return {'level': level, 'target_probability': prediction_proba, 'class_probability': probability}