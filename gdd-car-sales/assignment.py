import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

def read_data(filename):
    """
    Reads the CSV files; calculates features (age) and selects row & column subset
    """
    df = pd.read_csv(filename, encoding="ISO-8859-1")

    # Select subset of rows and columns
    df[(df.loc[:, "seller"] == "privat") & (df.loc[:, "offerType"] == "Angebot")]
    cols = [
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
    # Add only numerical columns
    df = df[cols]

    # Age column = (date of ad placement: extract only the year) - registration
    df["age"] = pd.to_datetime(df["dateCreated"]).dt.year - df["yearOfRegistration"]
    return df


if __name__ == "__main__":
    filename = pathlib.Path.cwd() / "data" / "autos.csv"
    df = read_data(filename)
    X_train, X_test = train_test_split(df, test_size=0.40, random_state=42)
    print(
        X_train.shape, X_test.shape,
    )
    print(X_train.head())
