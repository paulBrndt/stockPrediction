import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Data():
    def __init__(self, data, scaler: StandardScaler = StandardScaler()) -> None:
        self.value: pd.DataFrame
        self.scaler = scaler
        self.setupData(data)

    def setupData(self, input):
        self.initData(input)
        self.correctData()
        self.clearDataTypes()
        self.addData()
        self.clearDataTypes()
        self.declareFeatures()
        self.createTrainingsData()

    def initData(self, input):
        if isinstance(input, str):
            self.value = pd.read_csv(input)
        elif isinstance(input, pd.DataFrame):
            self.value = input
        else:
            raise TypeError("Data must be a string, or a pandas DataFrame")
        
    def correctData(self):
        self.value = self.value.dropna().drop_duplicates()

    def clearDataTypes(self):
        for col in self.value.columns.tolist():
            if col.lower() == "date":
                self.value[col] = self.value[col].astype("datetime64[ns]")

            elif self.value[col].dtype == "object":
                self.value[col]      = [row.replace("$", "") for row in self.value[col]]
                try:
                    self.value[col]  = self.value[col].astype("float64")
                except BaseException:
                    pass
            elif self.value[col].dtype == "int64":
                try:
                    self.value[col]      = self.value[col].astype("float64")
                except BaseException:
                    pass

    def addData(self):
        self.value["month"]             = self.value["Date"].dt.month
        self.value["isQuarterEnd"]      = np.where(self.value["month"] % 3==0, 1, 0)
        self.value["isYearEnd"]         = np.where(self.value["month"] % 12==0, 1, 0)

    def declareFeatures(self):
        self.value['open-close']        = self.value["Open"]     - self.value['Close']
        self.value['low-high']          = self.value['Low']      - self.value['High']
        self.value['target']            = np.where(self.value['Close'].shift(-1) > self.value['Close'], 1, 0) #wollen wir herausfinden, steigt oder sinkt Aktie

        #data setup
        self.features                   = self.value[["open-close", "low-high", "isQuarterEnd", "isYearEnd"]]
        self.features                   = self.scaler.fit_transform(self.features)
        self.target                     = self.value["target"]

    def createTrainingsData(self, testSize=0.1):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            self.features, self.target, test_size=testSize
        )
