# 📈 Stock Price Forecasting using Linear Regression

## 🔹 Overview

This project demonstrates a simple **time-series forecasting model** using **Linear Regression** to predict future stock prices. The model is trained on historical stock data and predicts the next few values based on past trends.

---

## 🔹 Features

* Data preprocessing with scaling
* Time-based label creation using shifting
* Sequential train-test split (no data leakage)
* Future price prediction
* Simple and easy-to-understand pipeline

---

## 🔹 Technologies Used

* Python 🐍
* NumPy
* Pandas
* Scikit-learn

---

## 🔹 How It Works

### 1. Data Preparation

* Load stock data from `prices.csv`
* Filter data for a specific company (GOOG)
* Create future labels using `.shift()`

```python
label = df[target_col].shift(-target_out)
```

---

### 2. Feature Engineering

* Convert data to NumPy array
* Scale features using `StandardScaler`

```python
X = StandardScaler().fit_transform(x)
```

---

### 3. Train-Test Split (Time-Series Aware)

* Instead of random split, data is split sequentially:

```python
split_index = int(len(X) * (1 - test_size))
```

---

### 4. Model Training

* Train a Linear Regression model on historical data:

```python
model = LinearRegression()
model.fit(x_train, y_train)
```

---

### 5. Prediction

* Predict future stock prices using last known data:

```python
prediction = model.predict(x_lately)
```

---

## 🔹 Output

* `prediction` → Forecasted stock prices for next `n` days
* `n_column.tail(5)` → Last known actual values (for comparison)

---

## 🔹 Example Output

```
Predicted Prices:
[1345.2, 1350.6, 1358.1, 1362.4, 1370.9]

Last Known Prices:
[1338.5, 1342.0, 1348.3, 1351.7, 1359.2]
```

---

## 🔹 Project Structure

```
├── prices.csv
├── main.py
└── README.md
```

---

## 🔹 Key Concepts Used

* Time Series Forecasting
* Feature Scaling
* Linear Regression
* Data Leakage Prevention
* Sequential Data Splitting

---

## 🔹 Limitations

* Assumes linear relationship in stock prices
* Does not handle seasonality or trends explicitly
* Basic model (can be improved using advanced techniques)

---

## 🔹 Future Improvements

* Use advanced models (LSTM, ARIMA)
* Add multiple features (volume, indicators)
* Hyperparameter tuning
* Visualization of predictions

---

## 🔹 How to Run

1. Install dependencies:

```bash
pip install numpy pandas scikit-learn
```

2. Run the script:

```bash
python main.py
```

---

## 🔹 Author

Developed as a learning project for understanding:

* Machine Learning pipelines
* Time-series data handling
* Real-world prediction workflow

---

## 🔹 License

This project is open-source and free to use.
