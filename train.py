import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score

from xgboost import XGBRegressor

import pickle

df = pd.read_csv("cars_24.csv")

# print(df.head())

# Extract 'Brand' from 'Name' Column
df["Brand"] = df["Name"].astype(str).str.split().str[0]

# Extract 'Model' from 'Name' Column
df["Model"] = df["Name"].astype(str).str.split().str[1:].str.join(" ")

df = df[["Name", "Brand", "Model", "Year", "Kms_Driven", "Owner", "Fuel", "Drive", "Type", "Price"]].copy()

# Create a backup of df
backup = df.copy()

# print(df.info())

# Drop the Missing Values
df.dropna(inplace = True)

# Drop duplicates
df.drop_duplicates(keep = "first", inplace = True)

# Standardize formatting for 'Brand' Column
df["Brand"] = df["Brand"].str.strip().str.upper()

# Cleaning 'Model' Column a bit:
df["Model"] = (
    df["Model"]
    .str.replace("-", " ", regex = False)
    .str.upper()
    .str.strip()
)

# Merging the models that are same(truly)
df["Model"] = df["Model"].replace({
    "SWIFT DZIRE": "DZIRE"
})

# Standardize formatting for 'Drive' and 'Type' Columns
df["Drive"] = df["Drive"].str.strip().str.upper()
df["Type"] = df["Type"].str.strip().str.upper()

# Change the dtypes of 'Year' and 'Owner Columns

df["Year"] = df["Year"].astype(int)
df["Owner"] = df["Owner"].astype('category')

# df.info()

# print(df.shape)

# Encode Rare Categories in 'Model' Column to 'Other'
model_counts = df["Model"].value_counts()
rare_models = model_counts[model_counts < 20].index
df["Model"] = df["Model"].apply(lambda x: "OTHER" if x in rare_models else x)

# Data Type Adjustments
df["Brand"] = df["Brand"].astype("category")
df["Model"] = df["Model"].astype("category")
df["Fuel"] = df["Fuel"].astype("category")
df["Drive"] = df["Drive"].astype("category")
df["Type"] = df["Type"].astype("category")

# df.to_csv("cleaned_cars_24.csv", index = False)

## Feature Engineering

# Creating the Car_Age Column from Year
current_year = datetime.now().year
df["Car_Age"] = current_year - df["Year"]

# Normalize the Kms_Driven Column by Year
df["Kms_per_Year"] = df["Kms_Driven"] / df["Car_Age"].replace(0, 1)

# Binarize the Owner Column
df["Is_First_Owner"] = (df["Owner"] == 1).astype(int)

# Frequency Encode the Model Column
model_freq = df["Model"].value_counts().to_dict()
df["Model_Freq"] = df["Model"].map(model_freq)

## Data Splitting

X = df[["Brand", "Model", "Model_Freq", "Car_Age", "Kms_per_Year", "Owner", "Is_First_Owner", "Fuel", "Drive", "Type"]]
y = df["Price"]
y_log = np.log1p(df["Price"])

X_train, X_test, y_train, y_test, y_log_train, y_log_test = train_test_split(X, y, y_log, test_size = 0.2, random_state = 39)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

## Data Preprocessing

# Preprocessing Pipelines

# Columns by type
log_cols = ["Kms_per_Year"]
scale_only_cols = ["Model_Freq", "Car_Age"]
cat_cols = ["Brand", "Model", "Owner", "Fuel", "Drive", "Type"]

# Log + Scaling Pipeline
log_pipeline = Pipeline([
    ("log", FunctionTransformer(np.log1p, feature_names_out = "one-to-one")),
    ("scale", StandardScaler())
])

# Scaling only Pipeline
scale_pipeline = Pipeline([
    ("scale", StandardScaler())
])

# Full Preprocessing Pipeline
full_preprocessing = ColumnTransformer(transformers = [
    ("log_num", log_pipeline, log_cols),
    ("scale_num", scale_pipeline, scale_only_cols),
    ("cat", OneHotEncoder(handle_unknown = "ignore"), cat_cols)
], remainder = "passthrough")

## Modelling

Xgb_Regressor = Pipeline([
    ("preprocessing", full_preprocessing),
    ("regressor", XGBRegressor(n_estimators = 200, max_depth = 7, learning_rate = 0.05, subsample = 0.8, colsample_bytree = 0.8, reg_alpha = 0, reg_lambda = 5, random_state = 39))
])

Xgb_Regressor.fit(X_train, y_log_train)
# y_pred = Xgb_Regressor.predict(X_test)
# y_true = np.expm1(y_log_test)
# y_pred = np.expm1(y_pred)
#
# r2 = r2_score(y_true, y_pred)
# rmse = root_mean_squared_error(y_true, y_pred)
#
# print(f"XGBoost Model → R² Score: {r2:.4f}, RMSE: {rmse:.2f}")

## Save the Model

pickle.dump(Xgb_Regressor, open("Xgb_Regressor.pkl", "wb"))