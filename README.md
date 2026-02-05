# PRODIGY ML Task-01: House Price Prediction

This project is part of the **PRODIGY InfoTech Machine Learning Internship (Task-01)**.

A **Linear Regression model** is used to predict house sale prices based on:
- Living area
- Number of bedrooms
- Total bathrooms (engineered feature)

The model is trained on `train.csv`, evaluated using standard regression metrics, and used to generate predictions for `test.csv`.

## Files
- `task1.py` – Model training and prediction code
- `train.csv` – Training dataset
- `test.csv` – Test dataset
- `submissions.csv` – Predicted sale prices

## How to Run
```bash
pip install pandas numpy matplotlib scikit-learn
python task1.py

