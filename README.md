# Used Car Price Predictor

An end-to-end machine learning project that predicts the re-sale price of used cars based on features like brand, fuel type, transmission, and kilometers driven. The model is trained to deliver reliable and consistent price estimates using regression techniques, helping individuals and dealerships make informed pricing decisions.

---

## Objective

This project uses machine learning to predict used car prices by analyzing historical data. It examines how features like manufacturing year, fuel type, body type, and ownership history impact pricing. The regression model is built through data cleaning, feature engineering, model training, and evaluation — for more accurate and fair pricing.

---

## Dataset Overview

The dataset contains detailed records of used cars, with features that influence resale value. It includes 8 input variables and 1 target variable (`Price`), and each row represents a unique car listing.

**Features:**
- `Car Name` – The name or model of the car.
- `Year Bought` – The year when the car was purchased.
- `Distance` – The distance already travelled by the car (in kilometres).
- `Previous Owners` – The number of previous owners of the car.
- `Fuel` – Type of fuel used (Petrol, Diesel, or CNG).
- `Location` - The location of the Regional Transport Office (RTO).
- `Car Type` – The type of car (sedan, SUV, hatchback, luxury SUV, luxury sedan).
- `Transmission` - The type of transmission (automatic or manual).

**Target:**
- `Price` – The price of the car. (in lakhs ₹)

This structured dataset is ideal for a supervised regression problem and requires minimal cleaning, making it suitable for end-to-end modeling.

[**Click here to view/download the dataset**](https://www.kaggle.com/datasets/ujjwalwadhwa/cars24com-used-cars-dataset)

---

## Exploratory Data Analysis

To uncover meaningful patterns and relationships within the data, here are some of the key visualizations used:

### Distribution of Price
> A histogram showed that most of the cars are priced below ₹10 lakh, with a noticeable right skew. This helped identify the presence of outliers and informed the need for potential scaling or transformation.

![](https://github.com/user-attachments/assets/73cab607-f7e9-4e94-9fa5-5ed64c355553)

