import pandas as pd

class MLPClassifier:
    def __init__(self, fileLocation):
        self.df = pd.read_csv(fileLocation)
    
    def run(self):
        pass