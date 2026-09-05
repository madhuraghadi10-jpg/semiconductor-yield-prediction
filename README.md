# 🔬 Semiconductor Yield Prediction

## 📌 Project Overview

This project uses Machine Learning to predict the Pass/Fail yield of a semiconductor manufacturing process using sensor measurements.

The project also investigates whether all available sensor features are required for building an effective prediction model.

---
## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Class Imbalance:** SMOTE
- **Model Optimization:** GridSearchCV, Cross-Validation
- **Model Deployment:** Streamlit
- **Model Persistence:** Joblib

## ✨ Key Features

- Predicts semiconductor manufacturing Pass/Fail outcomes
- Handles missing sensor values
- Removes highly incomplete features
- Performs exploratory data analysis
- Handles class imbalance using SMOTE
- Compares Logistic Regression, Random Forest, and SVM
- Uses Cross-Validation and GridSearchCV for model optimization
- Automatically selects the best-performing model
- Provides an interactive Streamlit prediction dashboard
- Displays model performance and feature importance

## 🔄 How It Works

1. Load semiconductor sensor data
2. Explore the dataset and target distribution
3. Remove highly incomplete features
4. Handle remaining missing values
5. Split the data into training and testing sets
6. Balance the training data using SMOTE
7. Standardize features where required
8. Train multiple classification models
9. Optimize models using Cross-Validation and GridSearchCV
10. Compare model performance
11. Select and save the best model
12. Use the trained model through the Streamlit dashboard


## 📈 Project Results

The trained models were evaluated using test accuracy and other classification metrics.

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 83.5% |
| Random Forest | 93.0% |
| SVM | 93.2% |

**Best Model:** SVM  
**Best Test Accuracy:** 93.2%

The model comparison shows that SVM achieved the highest test accuracy among the three evaluated models.

## 🔮 Future Improvements

- Improve model performance using advanced feature selection techniques
- Experiment with additional machine learning algorithms
- Apply dimensionality reduction techniques such as PCA
- Perform more extensive hyperparameter optimization
- Improve the Streamlit dashboard with additional visualizations
- Deploy the application to a cloud platform


## 🖥️ Application Preview

![Semiconductor Yield Prediction Dashboard](screenshots/app_dashboard.png)

## 🎯 Objective

The main objectives are:

- Predict Pass/Fail yield of a semiconductor manufacturing process.
- Clean and preprocess sensor data.
- Analyze the sensor measurements statistically.
- Handle missing values.
- Handle class imbalance.
- Compare different Machine Learning models.
- Perform cross-validation.
- Apply GridSearch hyperparameter tuning.
- Identify important sensor features.
- Select and save the best-performing model.

---

## 📊 Dataset

The dataset contains:

- 1,567 production entities
- 591 sensor/process features
- 1 target column
- Target: `Pass/Fail`

Target encoding:

| Value | Meaning |
|---|---|
| `-1` | Pass |
| `1` | Fail |

The original dataset contains 592 columns in total.

The project uses semiconductor sensor data containing 591 sensor features and a `Pass/Fail` target.

The full dataset is not included in this repository because of its size. A small `sample_input.csv` file is provided to demonstrate predictions through the Streamlit dashboard.

To train the models from scratch, place the original dataset at:

```text
data/sensor-data.csv

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Removed the timestamp column.
2. Identified features with excessive missing values.
3. Removed features with more than 50% missing values.
4. Imputed remaining missing values using the median.
5. Separated predictors and target.
6. Converted the target:
   - `-1 → 0` (Pass)
   - `1 → 1` (Fail)
7. Performed stratified train-test splitting.
8. Standardized features where required.
9. Applied SMOTE only to the training data.

---

## 📈 Exploratory Data Analysis

The analysis includes:

### Univariate Analysis

- Feature distributions
- Histograms
- Boxplots
- Detection of potential outliers

### Bivariate Analysis

- Sensor features vs Pass/Fail
- Correlation with target
- Feature distribution comparison

### Multivariate Analysis

- Correlation matrix
- Sensor-to-sensor relationships
- Correlation heatmap

---

## 🤖 Machine Learning Models

Three supervised learning algorithms were evaluated:

### 1. Logistic Regression

Used as a baseline linear classification model.

### 2. Random Forest

An ensemble tree-based model capable of capturing nonlinear relationships.

### 3. Support Vector Machine

Used to identify complex decision boundaries between Pass and Fail classes.

---

## ⚙️ Model Optimization

The models were evaluated using:

- 5-fold Cross Validation
- GridSearchCV
- Hyperparameter tuning
- SMOTE
- Standardization
- Classification metrics

---

## 📊 Model Evaluation

The models were compared using:

- Training Accuracy
- Testing Accuracy
- Cross-Validation Accuracy
- ROC-AUC
- Precision
- Recall
- F1-score
- Confusion Matrix

The final model was selected based on its performance on unseen test data.

---

## 🔎 Feature Importance

Permutation Importance was used to identify sensor features that contribute most to model predictions.

This helps identify potentially important process signals associated with Pass/Fail yield.

---

## 🏆 Final Model

The best model is selected automatically based on the highest test accuracy obtained during model comparison.

The trained model is saved as:

```text
models/best_model.pkl
```

## 📊 Model Performance

Three machine learning models were trained and compared:

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 83.5% |
| Random Forest | 93.0% |
| SVM | 93.2% |

### Best Model

**SVM** achieved the highest test accuracy of approximately **93.2%** and was selected as the best-performing model.

The models were evaluated using training/test accuracy, cross-validation, classification reports, and ROC-AUC.

## 📝 Conclusion

The project successfully developed a machine learning pipeline for semiconductor manufacturing yield prediction.

The dataset was cleaned by removing highly incomplete features and handling remaining missing values. Exploratory data analysis was performed to understand feature relationships and target distribution.

Three classification models were trained and evaluated: Logistic Regression, Random Forest, and SVM. After comparing their performance, **SVM achieved the highest test accuracy of approximately 93.2%** and was selected as the best-performing model.

The project demonstrates how machine learning can be used to analyze semiconductor sensor data and predict manufacturing Pass/Fail outcomes.

## 📄 License

This project is created for educational and portfolio purposes.