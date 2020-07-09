import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


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

# Joost: subset of the above
numerical_columns = []

def read_data(filename, columns_to_use):
    """
    Reads the CSV files; calculates features (age) and selects row & column subset
    """
    df = pd.read_csv(filename, encoding="ISO-8859-1")
    print(df.dateCreated)
    # Select subset of rows and columns
    df[(df.loc[:, "seller"] == "privat") & (df.loc[:, "offerType"] == "Angebot")]
    # Add only numerical columns
    df = df[columns_to_use]

    # Age column = (date of ad placement: extract only the year) - registration
    df["age"] = pd.to_datetime(df["dateCreated"]).dt.year - df["yearOfRegistration"]
    return df


def create_pipeline():
    """
    Creates a pipeline of all the prior steps.
    """

    # Pipeline from Joost
    num_pipeline = Pipeline(...)
    #
    #
    
    # Pipeline from Linsey
    cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine the two types of data:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", num_pipeline, numerical_columns),
            ("categorical", cat_pipeline, selector(dtype_include="category")),
        ],
        remainder="drop",
    )

    # Kevin: ColumnExtractor (from yesterday)
    # Eduard: RidgeRegression
    pipeline = Pipeline(
        steps=[
            ("select", ColumnExtractor(columns=columns_to_use)),
            ("preprocess", preprocessor),
            ("model", RidgeRegression()),
        ]
    )
    return pipeline


if __name__ == "__main__":

    df = read_data(filename, columns_to_use)
    
    X_train, X_test = train_test_split(df, test_size=0.40, random_state=42)

    categorical_transformer()
    print(
        X_train.shape, X_test.shape,
    )
    print(X_train.head())

    pipeline = create_pipeline()
    """
    Create a model pipeline that consists of ColumnTransformer for preprocessing the data and 
    RidgeRegression for fitting the model

    Perform a grid search over your whole pipeline and visualize the results. Right now your 
    pipeline does not have a whole lot of parameters to search over, you have an alpha in the 
    RidgeRegression and you can also play with an imputation strategy. If you have time and 
    energy left you can try adding other preprocessing steps (polynomial features, 
    different scalers, ...?) or a different ML model.
    """

    # Kevin: will set parameters to tune here
    param_grid = {
        "preprocess__numerical__impute__strategy": ["median", "mean"],
        "model__n_estimators": [10, 50, 100],
    }
    grid_clf = GridSearchCV(pipeline, param_grid, cv=5)
    grid_clf.fit(features, targets)

    # Kevin/others: visualize results
