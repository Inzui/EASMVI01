import pandas as pd
import numpy as np
from os.path import exists

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler

import pickle

class Classifier:
    #x0, y0, x1, y1, .. x21, y21, letter -> 43 columns
    def __init__(self):
        self.filename = "MLPModel"
        self.identifiers = ['A', 'D', 'E', 'I', 'L', 'N', 'O', 'R', 'S', 'T']
        self.scalerValues = None

    def run(self, data):
        """
            Load the model and scaler values from files if they exist. Normalize input data, predict the index and probability, and return the identifier and rounded probability.

            Args:
            - data: Input data to be normalized and predicted.

            Returns:
            - tuple: A tuple containing the identifier corresponding to the predicted index and the rounded probability of that index.

            Raises:
            - Exception: If no model file exists.
        """
        if (exists(self.filename)):
            self.model = pickle.load(open(self.filename, 'rb'))
            self.scalerValues = np.load("scalers.npy")
            
            normData = self.normalizeInputData(data)

            predictionIndex = self.model.predict([normData])[0]
            predictionProbability = self.model.predict_proba([normData])
            print(predictionProbability[0])
            return (self.identifiers[predictionIndex], round(predictionProbability[0][predictionIndex], 3))

        else:
            raise Exception("No Model has been created.")
        
    def normalizeInputData(self, data):
        retData = []
        for i in range(len(data)):
            retData.append(data[i] / self.scalerValues[i])
        return retData

    def train(self,  DFTrain : pd.DataFrame, DFTest : pd.DataFrame):
        """ 
            This method loads the training and testing datasets, cleans the data, splits the data into features (X) and labels (Y) for both training and testing sets. Then, it creates a model using these datasets and saves the model.

            Args:
            - DFTrain (DataFrame): The training dataset.
            - DFTest (DataFrame): The testing dataset.
            
            Raises:
            - Exception: If any error occurs during the process. 
        """

        self.DFTrain = DFTrain
        self.DFTest = DFTest
        self.cleanData()

        X_train = self.DFTrain.iloc[:,0:-1].values
        Y_train = self.DFTrain.iloc[:,-1].values

        X_test = self.DFTest.iloc[:,0:-1].values
        Y_test = self.DFTest.iloc[:,-1].values

        self.model = self.createModel(X_train, Y_train, X_test, Y_test)
        self.saveModel(self.model, self.filename)
        # print(self.getConfusionMatrix(self.model, X_val, Y_val))

    def createModel(self, X_train : pd.DataFrame, y_train : pd.DataFrame, X_test : pd.DataFrame, y_test : pd.DataFrame):
        """
            Create a Multi-Layer Perceptron (MLP) Classifier model.

            Args:
            - X_train: Training features.
            - y_train: Training labels.

            Returns:
            - mlp: Trained MLPClassifier model.
        """
        print("Creating Model")
        #mlp = MLPClassifier(activation='relu', solver='adam', random_state=1, hidden_layer_sizes=(550, 300), max_iter=1000, early_stopping=False, warm_start=False)
        mlp = MLPClassifier(activation='relu', solver='adam', random_state=1, hidden_layer_sizes=(1500, 1000), max_iter=500, early_stopping=False, warm_start=False)

        mlp.fit(X_train, y_train)
        print(f"Model Score: {mlp.score(X_test, y_test)}")
        return mlp
    
    def getConfusionMatrix(self, mlp : MLPClassifier, X_val : pd.DataFrame, y_val : pd.DataFrame):
        y_pred = mlp.predict(X_val)
        confusionMatrix = confusion_matrix(y_val, y_pred)
        return confusionMatrix
    
    def saveModel(self, mlp, filename):
        pickle.dump(mlp, open(filename, 'wb'))

    def cleanData(self):
        """ 
            This method cleans the data by converting the identifier column from letters to numbers, normalizing the data, and saving the scaler values for later use.

            Raises:
            - Exception: If the factorization fails due to invalid identifiers.
        """
        scaler = MaxAbsScaler()

        #Transfer identifier column from letter to number
        self.DFTrain["identifier"], _ = pd.factorize(self.DFTrain["identifier"])
        self.DFTest["identifier"], _ = pd.factorize(self.DFTest["identifier"])

        #Normalize all data
        scaler.fit(self.DFTrain)
        scaled = scaler.transform(self.DFTrain)
        self.DFTrain = pd.DataFrame(scaled, columns=self.DFTrain.columns)

        scaler.fit(self.DFTest)
        scaled = scaler.transform(self.DFTest)
        self.DFTest = pd.DataFrame(scaled, columns=self.DFTest.columns)

        np.save("scalers", scaler.scale_)

        self.DFTrain["identifier"], _ = pd.factorize(self.DFTrain["identifier"])
        self.DFTest["identifier"], _ = pd.factorize(self.DFTest["identifier"])    