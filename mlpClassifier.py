import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MaxAbsScaler

class Classifier:
    #x0, y0, x1, y1, .. x21, y21, letter -> 43 columns
    def __init__(self, DFTrain : pd.DataFrame, DFTest : pd.DataFrame):
        self.DFTrain = DFTrain
        self.DFTest = DFTest

    def run(self):
        self.cleanData()

        X_train = self.DFTrain.iloc[:,0:-1].values
        Y_train = self.DFTrain.iloc[:,-1].values

        X_val = self.DFTest.iloc[:,0:-1].values
        Y_val = self.DFTest.iloc[:,-1].values

        model = self.createModel(X_train, Y_train, X_val, Y_val)

    def createModel(self, X_train : pd.DataFrame, y_train : pd.DataFrame, X_val : pd.DataFrame, y_val : pd.DataFrame):
        mlp = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(100, 50), max_iter=10)
        mlp.fit(X_train, y_train)
        print(mlp.score(X_val, y_val))
        return mlp
    
    def getConfusionMatrix(self, mlp : MLPClassifier, X_val : pd.DataFrame, y_val : pd.DataFrame):
        y_pred = mlp.predict(X_val)
        confusionMatrix = confusion_matrix(y_val, y_pred)
        return confusionMatrix

    def cleanData(self):
        le = LabelEncoder()
        scaler = MaxAbsScaler()

        #Transfer identifier column from letter to number
        self.DFTrain["identifier"] = le.fit_transform(self.DFTrain["identifier"])
        self.DFTest['identifier'] = le.fit_transform(self.DFTest['identifier'])

        #Normalize all data
        scaler.fit(self.DFTrain)
        scaled = scaler.transform(self.DFTrain)
        self.DFTrain = pd.DataFrame(scaled, columns=self.DFTrain.columns)

        scaler.fit(self.DFTest)
        scaled = scaler.transform(self.DFTest)
        self.DFTest = pd.DataFrame(scaled, columns=self.DFTest.columns)
    