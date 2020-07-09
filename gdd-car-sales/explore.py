"""
https://towardsdatascience.com/custom-transformers-and-ml-data-pipelines-with-python-20ea2a7adb65


Download the dataset from Kaggle, 
https://www.kaggle.com/harlfoxem/housesalesprediction?select=kc_house_data.csv

and place it in the 'data' directory of this repo.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline

import numpy as np
import pandas as pd


data = pd.read_csv("data/kc_house_data.csv")


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom Transformer that extracts columns passed as argument to its constructor
    """

    def __init__(self, feature_names):
        self._feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self._feature_names]


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that breaks dates column into year, month and day into separate columns and
    converts certain features to binary
    """

    def __init__(self, use_dates=["year", "month", "day"]):
        self._use_dates = use_dates

    def fit(self, X, y=None):
        return self

    def get_year(self, obj):
        return str(obj)[:4]

    def get_month(self, obj):
        return str(obj)[4:6]

    def get_day(self, obj):
        return str(obj)[6:8]

    def create_binary(self, obj):
        """
        Helper function that converts values to Binary depending on input
        """
        if obj == 0:
            return "No"
        else:
            return "Yes"

    def transform(self, X, y=None):
        """
        Depending on constructor argument break dates column into specified units
        using the helper functions written above
        """
        for spec in self._use_dates:
            exec("X.loc[:,'{}'] = X['date'].apply(self.get_{})".format(spec, spec))

        # Drop unusable column
        X = X.drop("date", axis=1)

        # Convert these columns to binary for one-hot-encoding later
        X.loc[:, "waterfront"] = X["waterfront"].apply(self.create_binary)

        X.loc[:, "view"] = X["view"].apply(self.create_binary)

        X.loc[:, "yr_renovated"] = X["yr_renovated"].apply(self.create_binary)
        return X.values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer we wrote to engineer features (bathrooms per bedroom and/or how old the
    house is in 2019) passed as boolen arguements to its constructor.
    """

    def __init__(self, bath_per_bed=True, years_old=True):
        self._bath_per_bed = bath_per_bed
        self._years_old = years_old

    # Return self, nothing else to do here
    def fit(self, X, y=None):
        return self

    # Custom transform method we wrote that creates aformentioned features and drops redundant ones
    def transform(self, X, y=None):

        if self._bath_per_bed:
            # create new column
            X.loc[:, "bath_per_bed"] = X["bathrooms"] / X["bedrooms"]
            # drop redundant column
            X.drop("bathrooms", axis=1)

        if self._years_old:
            # create new column
            X.loc[:, "years_old"] = 2019 - X["yr_built"]
            # drop redundant column
            X.drop("yr_built", axis=1)

        # Converting any infinity values in the dataset to Nan
        X = X.replace([np.inf, -np.inf], np.nan)

        return X.values


# Categrical features to pass down the categorical pipeline
categorical_features = ["date", "waterfront", "view", "yr_renovated"]

# Numerical features to pass down the numerical pipeline
numerical_features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "condition",
    "grade",
    "sqft_basement",
    "yr_built",
]

# Defining the steps in the categorical pipeline
categorical_pipeline = Pipeline(
    steps=[
        ("cat_selector", FeatureSelector(categorical_features)),
        ("cat_transformer", CategoricalTransformer()),
        ("one_hot_encoder", OneHotEncoder(sparse=False)),
    ]
)
# Defining the steps in the numerical pipeline
numerical_pipeline = Pipeline(
    steps=[
        ("num_selector", FeatureSelector(numerical_features)),
        ("num_transformer", NumericalTransformer()),
        ("imputer", SimpleImputer(strategy="median")),
        ("std_scaler", StandardScaler()),
    ]
)

# Combining numerical and categorical piepline into one full big pipeline horizontally
# using FeatureUnion
full_pipeline = FeatureUnion(
    transformer_list=[
        ("categorical_pipeline", categorical_pipeline),
        ("numerical_pipeline", numerical_pipeline),
    ]
)


# Leave it as a dataframe becuase our pipeline is called on a
# pandas dataframe to extract the appropriate columns, remember?
X = data.drop("price", axis=1)
# You can covert the target variable to numpy
y = data["price"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The full pipeline as a step in another pipeline with an estimator as the final step
full_pipeline_m = Pipeline(steps=[("full_pipeline", full_pipeline), ("model", LinearRegression())])

# Can call fit on it just like any other pipeline
full_pipeline_m.fit(X_train, y_train)

# Can predict with it like any other pipeline
y_pred = full_pipeline_m.predict(X_test)
error = y_pred - y_test
print(error.describe())
