# Car-Price-Prediction
This is a Machine Learning project that predicts the **selling price of a used car** based on various features like Year, Fuel Type, Seller Type, Transmission, etc.

## 🔍 Project Overview

- **Dataset:** carData.csv
- **Models Used:**
  - Linear Regression (tested)
  - Lasso Regression (tested)
  - ✅ XGBoost Regressor (final model)
- **Evaluation Metrics:** R² Score, Scatter Plots (Actual vs Predicted)

## 📊 Features Used

- `Year`
- `Present_Price`
- `Kms_Driven`
- `Fuel_Type` (Petrol/Diesel/CNG)
- `Seller_Type` (Dealer/Individual)
- `Transmission` (Manual/Automatic)
- `Owner` (Number of previous owners)

## 📈 Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- xgboost

## ⚙️ How to Run

1. Clone the repository or download the code.
2. Make sure you have `carData.csv` in the same directory.
3. Run the Python script:
   ```bash
   python car_price_prediction.py
