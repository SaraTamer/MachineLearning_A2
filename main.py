import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(filename):
    data = pd.read_csv(filename)
    return data

# Task1: preprocessing
def missingData(data):
    # calculate number of missing
    missing_data = data.isnull().sum()
    # choose only columns include missing data
    missing_data = missing_data[missing_data > 0]
    print(missing_data)

def DropMissing(data):
    DataDropped=data.dropna()
    print("Data after dropping the missing :- ")
    print(DataDropped)


def ReplaceMissing(data):
    DataMiss = data.drop(columns=["Rain"])
    # fill the missing with mean of each feature
    FilledData = DataMiss.fillna(DataMiss.mean())
    FilledData["Rain"] = data["Rain"]
    print("Data after filling the missing :- ")
    print(FilledData)

def ScaleData(data):
    numeric_features = data.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    ScaledFeature = scaler.fit_transform(data[numeric_features])
    # ScaledFeature["Rain"] = data["Rain"]
    print(ScaledFeature)


def checkScalability(data):
    print(data.describe())
    MinMax=[]
    for column in data.select_dtypes(include=[np.number]):
        MinMax.append([data[column].min(), data[column].max()])

def SplitData(data):
    X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=0)

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )







def main():
    # a) Load the dataset
    data = load_dataset("C:/Users/EGYPT/PycharmProjects/ML_Assignment2/weather_forecast_data.csv")

    # missingData(data)
    # print("-----------------------------------------")
    # DropMissing(data)
    # print("-----------------------------------------")
    # ReplaceMissing(data)
    ScaleData(data)

if __name__ == "__main__":
    main()
