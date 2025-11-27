# 🌾 Pest Outbreak Prediction Using Machine Learning

This project explores a wide range of Machine Learning and Deep Learning methods to predict **crop disease severity one week ahead**, using agricultural data from Jordbruksverket and weather data from SMHI. The goal is to support farmers and authorities with early warnings that can reduce pesticide use, improve yield, and create more sustainable crop protection strategies.

---

## 📌 Overview

From 2016 to 2023, weekly Swedish crop health observations were combined with meteorological data to build a comprehensive dataset covering multiple crops and pests:

### **Crops**
- Höstvete  
- Rågvete  
- Vårkorn  

### **Pests**
Including (but not limited to):
- Bladfläcksvampar  
- Brunrost  
- Svartpricksjuka  
- Gulrost  
- Mjöldagg  
- Sköldfläcksjuka  

The project evaluates the forecasting performance of several ML and DL models:

- **HistGradientBoostingRegressor (best performing)**
- XGBoost Regressor
- Support Vector Regressor (SVR)
- Feedforward Neural Networks (FFNN)
- Transformer & LSTM architectures (experimental)
- Decision Tree Regressor (baseline)

---

## 🌟 Key Findings

- **HistGradientBoostingRegressor achieved the strongest predictive performance**, with an R² up to **0.88** for certain crop–pest combinations.
- **Lagged target features** (the previous week’s pest severity) were the most influential predictors across all datasets.
- **Cumulative weather metrics** (e.g., total seasonal rainfall, cumulative sunlight) significantly improved model accuracy.
- Model performance varied by dataset size and pest complexity—larger datasets consistently boosted accuracy.

Example results for the best model (HGBR):

| Crop     | Pest               | R²   | MAE  | MSE   |
|----------|--------------------|------|------|--------|
| Höstvete | Gulrost            | 0.88 | 0.89 | 11.79 |
| Höstvete | Svartpricksjuka    | 0.85 | 4.37 | 58.71 |
| Vårkorn  | Sköldfläcksjuka    | 0.60 | 0.55 | 4.81  |

---

## 📊 Data & Features

### **Data Sources**
- **Jordbruksverket (Prognos & Varnings API)**  
  Provides weekly pest and disease observations.
- **SMHI Meteorological API**  
  Provides weather data such as air temperature, precipitation, humidity, sunshine duration, dewpoint, and irradiance.

### **Feature Engineering**
- Weekly aggregation of raw agricultural & weather data  
- Previous-week lag values  
- Cumulative yearly weather parameters  
- Series identifiers for each field  
- Derived correlations between pests  
- Scaling and normalization based on historical extremes  

---

## 🧠 Modeling Approach

### **Models Evaluated**
- Gradient Boosting (HGBR, XGBR)
- Support Vector Regression (RBF kernel)
- Feedforward Neural Networks
- LSTM & Transformer networks (prototype stage)
- Decision Tree Regressor (baseline reference)

### **Training Method**
- **10-fold cross-validation** grouped by field (Series ID)  
- identical splits across model families for fairness  
- tuned hyperparameters for FFNN, SVR, and others  

### **Evaluation Metrics**
- **R²** – variance explained  
- **MAE** – average prediction error  
- **MSE** – punishes larger errors, important for peak outbreaks  

---

## 🚀 Impact & Use Cases

Accurate short-term forecasts enable:

- Earlier pest control decisions  
- Reduced pesticide usage  
- Lower environmental impact  
- Protection of crop yield  
- Data-driven planning for agricultural authorities  

A model that captures rising pest levels even **one week earlier** can make the difference between minimal intervention and significant crop damage.

---

## 🔮 Future Work

The project report outlines several improvements:

- Use weather data with **finer spatial resolution** (e.g., SMHI MESAN grid)
- Estimate **leaf wetness duration**, an important variable not directly available
- Reintroduce crop-strain sensitivity features once missing-data issues are resolved
- Apply **kriging** or other spatial interpolation methods to reduce station-distance errors
- Develop outlier-aware modeling strategies for peak pest values
- Train models directly on **forecasted** weather to simulate real deployment conditions

---

## 👥 Authors

Mahmut Osmanovic  
Isac Paulsson  
Sebastian Tuura  
Simon De Reuver  
Ivo Österberg Nilsson  

