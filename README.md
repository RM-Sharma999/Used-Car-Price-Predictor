# Used Car Price Predictor

An **end-to-end machine learning project** that predicts the resale price of used cars based on features like **brand, fuel type, transmission, and kilometers driven**. It uses **regression techniques** to build a **reliable and consistent model**, helping individuals and dealerships make **informed pricing decisions**.

---

## Objective

This project uses **machine learning** to predict used car prices by analyzing **historical data**. It examines how features like **manufacturing year, fuel type, body type, and ownership history** impact pricing. The **regression model** is built through **data cleaning, feature engineering, model training, and evaluation** to deliver more accurate and fair pricing.

---

## Dataset Overview

The dataset contains detailed records of used cars, with features that influence **resale value**. It includes **8 input variables** and **1 target variable (`Price`)**, and each row represents a unique car listing.

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

This structured dataset is ideal for a **supervised regression problem** and requires minimal cleaning, making it suitable for **end-to-end modeling**.

[**Click here to view/download the dataset**](https://www.kaggle.com/datasets/ujjwalwadhwa/cars24com-used-cars-dataset)

---

## Exploratory Data Analysis

To uncover **meaningful patterns and relationships** within the data, the following **key visualizations** were used:

### Distribution of Price
> This histogram showed that most cars are priced below **₹10 lakh**, with a noticeable **right skew**. This helped identify the presence of **outliers** and informed the need for potential scaling or transformation.

<img width="850" height="470" alt="image" src="https://github.com/user-attachments/assets/b53b7ebe-ff42-4f31-a42e-bf9112fe0791" />

### Year vs Price
> This boxplot revealed that **newer cars tend to retain higher resale value**, while **older vehicles depreciate significantly**. This confirmed the strong influence of manufacturing year on price.

<img width="1001" height="547" alt="image" src="https://github.com/user-attachments/assets/b5b77df5-b4b9-45e4-b8ee-32ba5f389165" />

### Kms Driven vs Price
> This scatter plot highlighted a **negative correlation** between kilometers driven and price. **Cars with higher mileage generally sell for less**, with a few **outliers** at both extremes.

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/7763ee7f-6b96-42fe-b58c-38f2a196fa2e" />

### Top 10 Brands and Models
> This subplot featuring bar charts displayed the **top 10 most listed car brands and models** in the dataset. This provided a clear overview of the most represented brands and models in the used car listings, helping reveal **supply trends** and **potential biases** in price predictions.

<img width="1483" height="489" alt="image" src="https://github.com/user-attachments/assets/5bf103a0-b094-41c8-b32b-a539f6cff187" />

### Owner Type vs Price
> This boxplot showed a **clear drop in resale value** as the number of previous owners increased. **Cars with a single owner typically retained higher prices**, highlighting how ownership history affects buyer perception and trust in vehicle condition.

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/be31d93c-abbd-4472-ad61-772f3bc404bc" />

---

## Data Preprocessing & Feature Engineering

The dataset was cleaned and prepared for **modeling**. The **Car_Name** column was split into **Brand** and **Model**, and several **new features** were created from existing columns. **Categorical variables** were one-hot encoded, and the final data was ready for training.

---

## Model Training & Evaluation

Multiple **regression models** were trained to predict used car prices. Their performance was evaluated using metrics like **Root Mean Square Error (RMSE)** and **R² Score** to identify the best-performing model.

### Models Used
- Linear Regression
- Ridge Regression
- K-Nearest Neighbors (KNN)
- Random Forest Regressor
- XGBoost Regressor

### Evaluation Metrics

- **RMSE:** Measures the average error in predictions. Lower is better.  
- **R² Score:** Shows how well the model explains the data. Higher is better.

### Performance Overview
The plot below compares the **performance of each model** based on **RMSE** and **R² Score**:

<img width="1189" height="560" alt="image" src="https://github.com/user-attachments/assets/00952447-dbd3-48e7-b6cb-dcd72e04a07c" />

---

## Technologies Used

- **Programming Language:** `Python`  
- **Visualization:** `Matplotlib`, `Seaborn` 
- **Machine Learning:** `Scikit-learn`, `Xgboost`  
- **Web Interface:** `Flask`  
- **Deployment Platform:** `Render`

---

## Deployment

The final model was deployed using **Flask** to create an **intuitive web-based interface**, and hosted on **Render** for public accessibility.

[Used Car Price Predictor Live App](https://used-car-price-predictor-dci0.onrender.com/)

---

## Key Takeaways

- **XGBoost** was the best-performing model with an **R² Score of 0.8465**, **cross-val R² Score of 0.8159**, and a **standard deviation of 0.0229**.
- Tuned multiple **regression models** to enhance performance and generalization using **hyperparameter optimization**.  
- Identified **key features** influencing used car prices using **XGBoost Feature Importance**.  
- Created a **user-friendly web app** using **Flask** and deployed it via **Render**.
