from typing import Callable

import numpy as np
import pickle as pk

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor

from utils import printNothing, infoMsg, Data

import datetime
import os
from dotenv import load_dotenv
load_dotenv()

#Fixme
#return as same unit as input (:

MODELS = [SVC, LogisticRegression, MLPRegressor]

class StockPredModel():
    def __init__(self, data, pathToSaved: str = "model.pkl", loadAtBeginning: bool = True, msg: Callable = None) -> None:

        self.data: list[Data]       = []
        self.msg                    = msg       or printNothing
        self.accuracy               = 0
        self.scaler                 = StandardScaler()

        if pathToSaved:
            self.pathToSaved = pathToSaved
            if loadAtBeginning:
                self.load()
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
        self.msg("trainiert")
        self.model.fit(self.xTrain, self.yTrain)
        self.getAccuracy()
        if self.accuracy > 0.5:
            self.save()

    def getAccuracy(self):
        self.accuracy = self.model.score(self.xTest, self.yTest)
        return self.accuracy

    def predict(self, data):
        return self.model.predict(data)
    
    def save(self):
        with open(f"models/{self.pathToSaved}", "wb") as f:
            print(self.pathToSaved)
            pk.dump(self, f)

    def load(self):
        with open(f"models/{self.pathToSaved}", "rb") as f:
            self.__dict__.update(pk.load(f).__dict__)
            self.train()
            print("completed")
    
files = os.environ.get("stock_array").replace("'", "").replace(" ", "").split(",")                

model = StockPredModel(files[:5], msg=infoMsg)
print(model.accuracy)
model.train()
print(model.predict(np.array([[0.4, -0.7600002289, 0.0, 0.0]])))
oldTime = datetime.datetime.now()
for item in files:
    print(f"Verarbeitung von Datei Nummer {files.index(item)} mit dem Namen {item}")
    model.addData(item)
    print(f"Wahrscheinlichkeit nach Training bei Testdaten richtig zu liegen: {model.accuracy: .3f}")
    model.train()
    model.save()
    time = datetime.datetime.now()
    print(f"Benötigte Zeit um mit Daten zu trainieren: {time - oldTime}")
    oldTime = time
    print("Um das Prgramm zu stoppen, tippe in das Terminal und drücke ^c")