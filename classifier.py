import pandas as pd

class Classifier:
    def __init__(self, dataFrame: pd.DataFrame):
        self.df = dataFrame
    
    def run(self):
        print("RUN")