from typing import Callable

import numpy as np
import pickle as pk

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor

from utils import printNothing, Data

import datetime
import os
from dotenv import load_dotenv
load_dotenv()

#Fixme
#return as same unit as input (:

MODELS = [SVC, LogisticRegression, MLPRegressor]

class StockPredModel():
    def __init__(self, data=[], pathToSaved: str = "model.pkl", msg: Callable = None) -> None:

        self.model                  = None
        self.data: list[Data]       = []
        self.msg                    = msg       or printNothing
        self.accuracy               = 0
        self.scaler                 = StandardScaler()

        if pathToSaved:
            self.pathToSaved = pathToSaved
            self.load()
        else:
            self.pathToSaved = "model.pkl"

        self.addData(data)
        if self.model is None:
            self.model = SVC()


    def addData(self, data):
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
        self.msg("trainieren")
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
            print(f"Model mit dem Namen '{self.pathToSaved.split('.')[0]}' geladen")

files = ['data/ANEB-1.csv', 'data/ATSG-1.csv', 'data/ATSG-2.csv', 'data/ATSG-3.csv', 'data/ATSG-4.csv', 'data/ATSG-5.csv', 'data/ATSG-6.csv', 'data/AVDX-1.csv', 'data/BFI-1.csv', 'data/BFI-2.csv', 'data/BRY-1.csv', 'data/BRY-2.csv', 'data/CATY-1.csv', 'data/CATY-2.csv', 'data/CATY-3.csv', 'data/CATY-4.csv', 'data/CATY-5.csv', 'data/CATY-6.csv', 'data/CATY-7.csv', 'data/CATY-8.csv', 'data/CATY-9.csv', 'data/CCRN-1.csv', 'data/CCRN-2.csv', 'data/CCRN-3.csv', 'data/CCRN-4.csv', 'data/CCRN-5.csv', 'data/CCRN-6.csv', 'data/CDAQ-1.csv', 'data/CGBD-1.csv', 'data/CGBD-2.csv', 'data/CZFS-1.csv', 'data/CZFS-2.csv', 'data/CZFS-3.csv', 'data/CZFS-4.csv', 'data/CZFS-5.csv', 'data/CZFS-6.csv', 'data/CZFS-7.csv', 'data/DTI-1.csv', 'data/FDMT-1.csv', 'data/HKIT-1.csv', 'data/HistoricalQuotes-1.csv', 'data/HistoricalQuotes-2.csv', 'data/HistoricalQuotes-3.csv', 'data/IONM-1.csv', 'data/IONM-2.csv', 'data/LSTA-1.csv', 'data/LSTA-2.csv', 'data/LSTA-3.csv', 'data/LSTA-4.csv', 'data/LSTA-5.csv', 'data/LSTA-6.csv', 'data/LSTA-7.csv', 'data/NERV-1.csv', 'data/NERV-2.csv', 'data/NERV-3.csv', 'data/NHTC-1.csv', 'data/NHTC-2.csv', 'data/NHTC-3.csv', 'data/NHTC-4.csv', 'data/NHTC-5.csv', 'data/NHTC-6.csv', 'data/NHTC-7.csv', 'data/NHTC-8.csv', 'data/OLLI-1.csv', 'data/OLLI-2.csv', 'data/OLLI-3.csv', 'data/OP-1.csv', 'data/ORIC-1.csv', 'data/ORIC-2.csv', 'data/PAL-1.csv', 'data/PFTAU-1.csv', 'data/SNPX-1.csv', 'data/Tesla-1.csv', 'data/Tesla-2.csv', 'data/Tesla-3.csv', 'data/Tesla-4.csv', 'data/WBTN-1.csv', 'data/XELB-1.csv', 'data/XELB-2.csv', 'data/XELB-3.csv']
prextenison = os.environ.get("HOME_FOLDER")
files = [f"{prextenison}/{file}" for file in files]


def loadModel() -> StockPredModel:
    return StockPredModel()

def train(model: StockPredModel = StockPredModel()):
    oldTime = datetime.datetime.now()
    for item in files:
        print(f"Verarbeitung von Datei Nummer {files.index(item)} mit dem Namen {item}")
        model.addData(item)
        print(f"Wahrscheinlichkeit nach Training bei Testdaten richtig zu liegen: {model.accuracy*100: .1f}%")
        model.train()
        model.save()
        time = datetime.datetime.now()
        print(f"Benötigte Zeit um mit Daten zu trainieren: {time - oldTime}")
        oldTime = time
        print("Um das Prgramm zu stoppen, tippe in das Terminal und drücke ^c")
        print()


def vorhersagen():
    model = loadModel()
    print(model.accuracy)

train()
vorhersagen()