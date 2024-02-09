import pickle

class ComplevelPredictor():
    """
    A class that represents a ComplevelPredictor.

    This class provides methods to deserialize a trained model and make predictions based on input text.

    Attributes:
        None

    Methods:
        deserialize(): Deserialize the trained model.
        predict(title, description): Make a prediction based on the given title and description.

    """
  
    def deserialize(self):
        """
        Deserialize the trained model.

        Returns:
            model (object): The deserialized model.

        """
        with open('./data/models/comp-level_ai-model.pickle', 'rb') as handle:
            model = pickle.load(handle)
            return model
  
    def predict(self, title, description):
        """
        Make a prediction based on the given title and description.

        Args:
            title (str): The title of the input text.
            description (str): The description of the input text.

        Returns:
            dict: A dictionary containing the predicted level, target probability, and class probabilities.

        """
        text = title + " \n\n " + description
        model = self.deserialize()
        prediction = model.predict([text]).tolist()[0]
        probability = model.predict_proba([text]).tolist()[0]
        labels = ['A', 'B', 'C']
        level = prediction[0]
        prediction_proba = probability[labels.index(level)]
        return {'level': level, 'target_probability': prediction_proba, 'class_probability': probability}