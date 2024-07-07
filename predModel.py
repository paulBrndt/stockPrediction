from typing import Callable

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor

from utils import printNothing, infoMsg

#Fixme
#return as same unit as input (:

MODELS = [SVC, LogisticRegression, MLPRegressor]

class StockPredModel():
    def __init__(self, data, msg: Callable = None, debugMsg: Callable = None, model = MODELS[0]()) -> None:

        self.data       = data
        self.msg        = msg       or printNothing
        self.debugMsg   = debugMsg  or printNothing

        self.setupData()

        self.scaler     = StandardScaler()
        self.features   = self.scaler.fit_transform(self.features)

        self.model      = model

        self.setupTraining()


    def setupData(self):
        self.initData()
        self.correctData()
        self.clearDataTypes()
        self.addData()
        self.clearDataTypes()
        self.declareFeatures()
        self.msg("Data is setup")

    def initData(self):
        if isinstance(self.data, str):
            self.data   = pd.read_csv(self.data)
        elif isinstance(self.data, pd.DataFrame):
            self.data   = self.data
        else:
            raise TypeError("Data must be a string or a pandas DataFrame")

    def correctData(self):
        self.data = self.data.dropna()
        self.data = self.data.drop_duplicates()

    def clearDataTypes(self):
        for col in self.data.columns.tolist():
            print("-"*100)
            print(col, "\n")
            if col.lower() == "date":
                self.data[col] = self.data[col].astype("datetime64[ns]")

            elif self.data[col].dtype == "object":
                self.data[col]      = [row.replace("$", "") for row in self.data[col]]
                try:
                    self.data[col]  = self.data[col].astype("float64")
                except:
                    pass
            elif self.data[col].dtype == "int64":
                try:
                    self.data[col]      = self.data[col].astype("float64")
                except:
                    pass
    
    def addData(self):
        self.data["month"]              = self.data["Date"].dt.month
        self.data["isQuarterEnd"]       = np.where(self.data["month"] % 3==0, 1, 0)
        self.data["isYearEnd"]          = np.where(self.data["month"] % 12==0, 1, 0)

    
    def declareFeatures(self):
        self.data['open-close']         = self.data["Open"]     - self.data['Close/Last']
        self.data['low-high']           = self.data['Low']      - self.data['High']
        self.data['target']             = np.where(self.data['Close/Last'].shift(-1) > self.data['Close/Last'], 1, 0) #wollen wir herausfinden, steigt oder sinkt Aktie

        #data setup
        self.features                   = self.data[["open-close", "low-high", "isQuarterEnd", "isYearEnd"]]
        self.target                     = self.data["target"]



    def setupTraining(self):
        self.xTrain = self.xTest = self.yTrain = self.yTest = np.zeros((0, 0))
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.features, self.target, test_size=0.1
        )



    def train(self):
        print("running")
        self.model
        self.model.fit(self.xTrain, self.yTrain)
        print(f"{self.model}:")
        print(f"Accuracy: {self.model.score(self.xTest, self.yTest)}")
        print()

    #use network
    def predict(self, data):
        return self.model.predict(data)
    
                  





model = StockPredModel("path", infoMsg, infoMsg)
model.train()