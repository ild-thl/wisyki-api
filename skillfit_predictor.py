import pickle
import os

class skillfit_predictor():
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        self.optimal_threshold = 0.5
        # Load the saved scaler
        with open(self.dir + '/models/skillfit_scaler.pickle', 'rb') as file:
            self.scaler = pickle.load(file)
  
    def deserialize(self):
        with open(self.dir + '/models/skillfit_ai-model.pickle', 'rb') as handle:
            model = pickle.load(handle)
            return model
  
    def predict(self, X):
        model = self.deserialize()
        # Scale your new data
        X_scaled = self.scaler.transform(X)
        y_pred_probs = model.predict(X_scaled)
        return (y_pred_probs >= self.optimal_threshold).astype(int).tolist()