# Setup and imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import time
import pickle
import os
import json

class skillfit_model_trainer():
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        pass

    def getReport(self):
        if os.path.exists(self.dir + "/logs/skillfitReport.json"):
            fileObject = open(self.dir + "/logs/skillfitReport.json", "r")
            jsonContent = fileObject.read()
            return json.loads(jsonContent)
        else:
            return None
    

    def train(self):
        log = "Training results:"
        log += "\nDate: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # Start time.
        st = time.time()

        # Load dataset
        dataset = pd.read_json(self.dir + "/data/skillfit_dataset_notrack.json")

        report = {}
        report['dataset_size'] = dataset.shape[0]
        log += "\nTraining data size: " + str(dataset.shape[0])

        # Load your dataset (assuming it's in a pandas DataFrame)
        # Replace 'target_column_name' with the actual name of your target column
        target_column_name = 'fit'
        y = dataset[target_column_name]

        # Count the occurrences of each class
        class_counts = y.value_counts()

        # Compute class ratios
        total_samples = len(y)
        class_ratios = class_counts / total_samples

        log += "\n\nClass Ratios: " + str(class_ratios.tolist())

        # X is data with the column "fit" removed
        X_doc_embedding = dataset['doc_embedding']
        X_skill_embedding = dataset['skill_embedding']
        X_similarity = dataset['similarity']

        # Convert the embeddings to numpy arrays
        X_doc_embedding = np.array(X_doc_embedding.tolist())
        X_skill_embedding = np.array(X_skill_embedding.tolist())

        # Concatenate the embeddings and similarity into a single feature matrix
        X = np.concatenate((X_doc_embedding, X_skill_embedding, X_similarity.values.reshape(-1, 1)), axis=1)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Feature Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Save the scaler to a file
        with open('models/skillfit_scaler.pickle', 'wb') as file:
            pickle.dump(scaler, file)
        
        # Handle Class Imbalance using SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Define a parameter grid to search
        param_grid = {
            'input_dim': [X_train.shape[1]],
            'units': [64, 128],
            'activation': ['relu'],
            'dropout_rate': [0.2, 0.3],
        }        
        
        # Early Stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Create the KerasClassifier for use in GridSearchCV
        keras_model = KerasClassifier(build_fn=self.create_model, epochs=100, batch_size=128, verbose=0, validation_split=0.2, callbacks=[early_stopping])

        # Perform grid search
        grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3, verbose=2)
        grid_result = grid.fit(X_train, y_train)
        
        # Get the best model from grid search
        best_model = grid_result.best_estimator_
        
        # Save the best performing model.
        pickle.dump(best_model, open(self.dir + "/models/skillfit_ai-model.pickle", 'wb'))

        # Log the best parameters and results
        report['best_score'] = grid_result.best_score_
        report['best_hyperparameter'] = grid_result.best_params_
        log += "\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_)

        # Get predictions for evaluation dataset.
        y_pred_probs = best_model.predict(X_test)

        # Initialize variables for optimal threshold and max F1-score
        optimal_threshold = 0
        max_f1_score = 0

        # Initialize binary search range
        low, high = 0, 1

        # Set the number of iterations for the binary search
        num_iterations = 100

        # Perform binary search to find the optimal threshold
        for _ in range(num_iterations):
            threshold = (low + high) / 2
            y_pred = (y_pred_probs >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            if f1 > max_f1_score:
                max_f1_score = f1
                optimal_threshold = threshold
            if f1 < 0.5:
                high = threshold
            else:
                low = threshold

        # Use the optimal threshold for predictions
        y_pred = (y_pred_probs >= optimal_threshold).astype(int)

        log += "\n\nOptimal Threshold for Max F1-Score: " + str(optimal_threshold)
        log += "\n\nMax F1-Score: " + str(max_f1_score)

        # Evaluate the model
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        report['accuracy'] = accuracy
        log += "\n\nTest accuracy: {:.2f}%".format(accuracy * 100)
        # Precision
        precision = precision_score(y_test, y_pred)
        report['precision'] = precision
        log += "\n\nTest precision: {:.2f}%".format(precision * 100)
        # Recall
        recall = recall_score(y_test, y_pred)
        report['recall'] = recall
        log += "\n\nTest recall: {:.2f}%".format(recall * 100)
        # F1 Score
        f1 = f1_score(y_test, y_pred)
        report['f1'] = f1
        log += "\n\nTest f1: {:.2f}%".format(f1 * 100)

        # End time.
        et = time.time()

        # Get Elapsed time.
        elapsed_time = et - st
        log += "\n\nExecution time: " + str(elapsed_time) + ' seconds'
        print(log)

        report['time'] = st
        report['executiontime'] = elapsed_time
        report['modelname'] = 'skillfit-sequential'

        
        if os.path.exists(self.dir + "/logs/skillfitReport.json"):
            with open(self.dir + "/logs/skillfitReport.json", "r+") as jsonFile:
                try:
                    reports = json.load(jsonFile)
                except:
                    reports = []
                
                reports.append(report)
                jsonFile.seek(0)
                json.dump(reports, jsonFile)
                jsonFile.truncate()
        else:
            with open(self.dir + "/logs/skillfitReport.json", "w") as jsonFile:
                json.dump([report], jsonFile)

        # Check if "/logs/trainSkillfitModelLog.txt exists.
        if os.path.exists(self.dir + "/logs/trainSkillfitModelLog.txt"):
            logFile = open(self.dir + "/logs/trainSkillfitModelLog.txt", "a")
        else:
            logFile = open(self.dir + "/logs/trainSkillfitModelLog.txt", "w")

        logFile.write(log + "\n\n\n")
        logFile.close()

        return report
    

    # Create your model function
    def create_model(self, input_dim, units=64, activation='relu', dropout_rate=0.2):
        model = Sequential()
        model.add(Dense(units=units, activation=activation, input_dim=input_dim))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


