import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("data/cleaned_sensor_data.csv")


# ---------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(columns=["Pass/Fail"])

y = df["Pass/Fail"].map({
    -1: 0,
    1: 1
})


# ---------------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ---------------------------------------------------------
# 5. SMOTE
# ---------------------------------------------------------

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)


print("=" * 60)
print("RANDOM FOREST")
print("=" * 60)

print("\nTraining shape after SMOTE:")
print(X_train.shape)

print("\nClass distribution:")
print(pd.Series(y_train).value_counts())


# ---------------------------------------------------------
# 6. CREATE RANDOM FOREST
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# 7. CROSS VALIDATION
# ---------------------------------------------------------

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("\nCross-validation scores:")
print(cv_scores)

print(
    "\nMean CV accuracy:",
    cv_scores.mean()
)


# ---------------------------------------------------------
# 8. TRAIN MODEL
# ---------------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# 9. PREDICTIONS
# ---------------------------------------------------------

train_predictions = model.predict(X_train)

test_predictions = model.predict(X_test)


# ---------------------------------------------------------
# 10. ACCURACY
# ---------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

print("\nTraining Accuracy:")
print(train_accuracy)

print("\nTesting Accuracy:")
print(test_accuracy)


# ---------------------------------------------------------
# 11. CLASSIFICATION REPORT
# ---------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Pass", "Fail"]
    )
)


# ---------------------------------------------------------
# 12. CONFUSION MATRIX
# ---------------------------------------------------------

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)