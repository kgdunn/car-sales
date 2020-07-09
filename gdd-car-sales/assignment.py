import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Load from the data folder. Note: this script starts in the directory one level higher than the
# data folder.
filename = pathlib.Path.cwd() / "data" / "autos.csv"
columns_to_use = [
    "brand",
    "gearbox",
    "powerPS",
    "kilometer",
    "fuelType",
    "model",
    "notRepairedDamage",
    "yearOfRegistration",
    "dateCreated",
]

def convert_to_numerical(df):
    labelencoder_X = LabelEncoder()
    df[:,0] = labelencoder_X.fit_transform(X[:,0])
    onehotencoder = OneHotEncoder(categorical_features=[0,1])
    X = onehotencoder.fit_transform(X).toarray()
    



def read_data(filename, columns_to_use):
    """
    Reads the CSV files; calculates features (age) and selects row & column subset
    """
    df = pd.read_csv(filename, encoding="ISO-8859-1")

    # Select subset of rows and columns
    df[(df.loc[:, "seller"] == "privat") & (df.loc[:, "offerType"] == "Angebot")]
    # Add only numerical columns
    df = df[columns_to_use]

    # Age column = (date of ad placement: extract only the year) - registration
    df["age"] = pd.to_datetime(df["dateCreated"]).dt.year - df["yearOfRegistration"]
    return df


if __name__ == "__main__":

    df = read_data(filename, columns_to_use)
    print(convert_cat_to_num(df))
    X_train, X_test = train_test_split(df, test_size=0.40, random_state=42)
    print(
        X_train.shape, X_test.shape,
    )
    print(X_train.head())
