# Diabetes Risk Prediction — Multi-Class Classification

Predicting diabetes risk level (Low / Moderate / High) from demographic, lifestyle, and clinical health data using both Neural Network and XGBoost.

## Overview
This project builds and compares two multi-class classification models to predict a patient's diabetes risk category based on 18 features spanning demographics (age, gender, city), lifestyle (diet, smoking, alcohol, physical activity, sleep, stress), and clinical measurements (BMI, fasting blood sugar, HbA1c, blood pressure, waist circumference).

## Dataset
- 15,000 patient records, 18 features
- Target: `diabetes_risk` — Low (60%), Moderate (25%), High (15%)
- Missing values present in `alcohol_consumption`, `smoking_status`, `income_bracket`

## Approach
1. **Exploratory Data Analysis** — distributions, correlations, outlier detection, class imbalance check
2. **Missing Value Handling** — verified missingness was not correlated with the target (MCAR/MAR) before imputing with an explicit "Unknown" category, preserving information rather than fabricating values
3. **Preprocessing** — one-hot encoding, stratified train/CV/test split (70/15/15), feature scaling, class weighting for imbalance
4. **Modeling**
   - Neural Network (TensorFlow/Keras) 
   - XGBoost
5. **Evaluation** — accuracy, per-class precision/recall/F1, macro-F1, confusion matrices

## Results

| Metric | Neural Network (baseline) | XGBoost (baseline) |
|---|---|---|
| Test Accuracy | 77.5% | 76.0% |
| Low F1 | 0.86 | 0.85 |
| Moderate F1 | 0.60 | 0.59 |
| High F1 | 0.76 | 0.72 |
| Macro F1 | 0.74 | 0.72 |

**Key insight:** The "Moderate" risk class was consistently the hardest to classify for both models — it sits on a continuum between Low and High on clinical markers like `hba1c_level` and `fasting_blood_sugar`, making it inherently less separable than a distinct cluster.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn (preprocessing, splitting, metrics)
- TensorFlow / Keras
- XGBoost
- Matplotlib, Seaborn

