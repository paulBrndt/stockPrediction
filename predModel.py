from typing import Callable

import numpy as np
import pickle as pk

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor

from utils import printNothing, infoMsg, Data

import os
from dotenv import load_dotenv
load_dotenv()

#Fixme
#return as same unit as input (:

MODELS = [SVC, LogisticRegression, MLPRegressor]

class StockPredModel():
    def __init__(self, data, pathToSaved: str = "", loadAtBeginning: bool = False, msg: Callable = None) -> None:

        self.data: list[Data]       = []
        self.msg                    = msg       or printNothing
        self.accuracy               = 0
        self.scaler                 = StandardScaler()

        if pathToSaved:
            if loadAtBeginning: self.load(pathToSaved)
        else:
            self.pathToSaved = "model.pkl"

        self.addData(data)
        self.model                  = SVC()

    def addData(self, data):
        self._setupData(data)
        self._setupTraining()

    def _setupData(self, data):
        if isinstance(data, list):
            for item in data:
                self.data.append(Data(item))
        else:
            self.data.append(Data(data))

    def _setupTraining(self, testSize = 0.1):
        allFeatures = np.vstack([data.features for data in self.data])
        allTargets  = np.hstack([data.target for data in self.data])
        
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(allFeatures, allTargets, test_size=testSize)

    def train(self):
        self.msg("training")
        self.model.fit(self.xTrain, self.yTrain)
        self.getAccuracy()

    def getAccuracy(self):
        self.accuracy = self.model.score(self.xTest, self.yTest)
        return self.accuracy

    def predict(self, data):
        return self.model.predict(data)
    
    def save(self, filePath: str = ""):
        self.pathToSaved = filePath if filePath else self.pathToSaved
        with open(self.pathToSaved, "wb") as f:
            pk.dump(self, f)

    def load(self, filePath: str = ""):
        self.pathToSaved = filePath if filePath else self.pathToSaved
        with open(self.pathToSaved, "rb") as f:
            self.__dict__.update(pk.load(f).__dict__)
            self.train()
            print("completed")
    
                  

load_dotenv()

model = StockPredModel([os.environ.get("AAPL_PATH"), os.environ.get("TSLA_PATH")], pathToSaved="", msg=infoMsg)
print(model.accuracy)
model.train()
print(model.accuracy)
model.save()
model.train()
print(model.accuracy)