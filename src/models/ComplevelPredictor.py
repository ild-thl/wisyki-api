import pickle
from pydantic import BaseModel, Field, validator
from typing import List, Literal

class PredictCompLevelRequest(BaseModel):
    title: str = Field(default="", description="The title of the course.")
    description: str = Field(default="", description="The description of the course.")


class CompLevelResponse(BaseModel):
    class_probability: List[float] = Field(..., description="The probability of each class.")
    level: Literal["A", "B", "C", "D"] = Field(
        ..., description="The level must be one of 'A', 'B', 'C', or 'D'"
    )
    target_probability: float = Field(..., description="The probability of the target class.")

    @validator("level")
    def check_level(cls, v):
        if v not in ["A", "B", "C", "D"]:
            raise ValueError('level must be one of "A", "B", "C", or "D"')
        return v

class LegacyCompLevelResponse(BaseModel):
    class_probability: List[float] = Field(..., description="The probability of each class.")
    level: str = Field(
        ..., description="The level must be one of 'A', 'B', or 'C'"
    )
    target_probability: float = Field(..., description="The probability of the target class.")

    @validator("level")
    def check_level(cls, v):
        if v not in ["A", "B", "C"]:
            if v == "D":
                v = "C"
            else:
                raise ValueError('level must be one of "A", "B", or "C"')
        return v

    # Sum up the probabilities of levels "C" and "D" and return them as "C"
    @validator("class_probability")
    def sum_probabilities(cls, v):
        if len(v) == 4:
            v[2] = v[2] + v[3]
            v.pop()
        return v

    # If the level is "D", set the target probability to the probability of "C"
    @validator("target_probability")
    def set_target_probability(cls, v, values):
        if values["level"] == "D" or values["level"] == "C":
            v = values["class_probability"][2]


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
        labels = ['A', 'B', 'C', 'D']
        level = prediction[0]
        prediction_proba = probability[labels.index(level)]
        return {'level': level, 'target_probability': prediction_proba, 'class_probability': probability}