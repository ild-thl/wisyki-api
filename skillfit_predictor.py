import pickle
import os
from tensorflow import keras

class skillfit_predictor():
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        self.optimal_threshold = 0.5
        # Load the saved scaler
        self.scaler = pickle.load(open(self.dir + "/models/skillfit_ai/skillfit_scaler.pickle", "rb"))
        self.model = keras.models.load_model(self.dir + "/models/skillfit_ai/skillfit_ai-model")
  
    def predict(self, X):
        # Scale your new data
        X_scaled = self.scaler.transform(X)
        y_pred_probs = self.model.predict(X_scaled)
        return (y_pred_probs >= self.optimal_threshold).astype(int).tolist()