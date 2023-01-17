import pickle
import os

class topic_predictor():
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        pass
  
    def deserialize(self):
        with open(self.dir + '/data/topic_model.pickle', 'rb') as handle:
            model = pickle.load(handle)
            return model
  
    def predict(self, title, description):
        text = title + " \n\n " + description
        model = self.deserialize()
        prediction = model.predict([text]).tolist()[0]
        probability = model.predict_proba([text]).tolist()[0]
        labels = ['Wirtschaft, Büro, Management', 'Sprachen',
            'Technik, Logistik, Umwelt, Naturwissenschaften',
            'Computer-Administration, IT, Programmierung', 'EDV-Anwendung',
            'Gesellschaft, Politik, Studienreisen', 'Sozialwesen',
            'Kultur, Kunst, Medien, Mode', 'Persönliche und soziale Kompetenz',
            'Touristik, Gastronomie, Hauswirtschaft',
            'Schulabschlüsse, Studienvorbereitung']
        topic = prediction[0]
        prediction_proba = probability[labels.index(topic)]
        return {'topic': topic, 'target_probability': prediction_proba, 'class_probability': probability}