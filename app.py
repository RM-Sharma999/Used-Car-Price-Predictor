from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Formats the price in Indian Style
def format_in_indian_currency(number):
    number = float(number)
    integer_part = int(number)
    decimal_part = f"{number:.2f}".split(".")[1]

    s = str(integer_part)
    r = ""
    if len(s) > 3:
        r = "," + s[-3:]
        s = s[:-3]
        while len(s) > 2:
            r = "," + s[-2:] + r
            s = s[:-2]
    return f"{s}{r}.{decimal_part}"

# Load the Model
xgb_model = pickle.load(open("Xgb_Regressor.pkl", "rb"))

# Load the Cleaned Cars_24 dataset
car = pd.read_csv("cleaned_cars_24.csv")

# Create a model_freq dictionary
model_freq = car["Model"].value_counts().to_dict()

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def index():
    Brand_of_car = sorted(car["Brand"].unique())
    ModelDict = car.groupby("Brand")["Model"].unique().apply(list).to_dict()
    Year = sorted(car["Year"].unique())
    Fuel_Type = sorted(car["Fuel"].unique())
    Transmission_Type = sorted(car["Drive"].unique())
    Body_Type = sorted(car["Type"].unique())

    return render_template("index.html",
                           Brand = Brand_of_car,
                           ModelDict = ModelDict,
                           Year = Year,
                           Fuel = Fuel_Type,
                           Transmission = Transmission_Type,
                           Body = Body_Type)

@app.route("/predict", methods = ["POST"])
def predict():
    Brand = request.form.get("brand")
    Model = request.form.get("model")
    Year = int(request.form.get("year"))
    Fuel = request.form.get("fuel_type")
    Drive = request.form.get("transmission")
    Type = request.form.get("body_type")
    Owner = request.form.get("owner_type")
    Is_First_Owner = int(request.form.get("is_first_owner"))
    Kms_Driven = int(request.form.get("kms_driven"))

    # Feature Engineering
    Model_Freq = model_freq.get(Model, 0)

    current_year = datetime.now().year
    Car_Age = current_year - Year

    Kms_per_Year = Kms_Driven / (Car_Age if Car_Age != 0 else 1)

    input_df = pd.DataFrame(data = {"Brand": Brand,
                                    "Model": Model,
                                    "Model_Freq": Model_Freq,
                                    "Car_Age": Car_Age,
                                    "Kms_per_Year": Kms_per_Year,
                                    "Owner": Owner,
                                    "Is_First_Owner": Is_First_Owner,
                                    "Fuel": Fuel,
                                    "Drive": Drive,
                                    "Type": Type}, index = [0])

    # Predict log(price)
    log_prediction = xgb_model.predict(input_df)

    # log(price) -> price
    predicted_price = np.expm1(log_prediction[0])

    return format_in_indian_currency(predicted_price)


if __name__ == "__main__":
    app.run(debug = True)
