import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("data/cleaned_sensor_data.csv")

print("=" * 60)
print("STANDARDIZATION")
print("=" * 60)


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
# 4. HANDLE ANY REMAINING MISSING VALUES
# ---------------------------------------------------------

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)

X_test = imputer.transform(X_test)


# ---------------------------------------------------------
# 5. STANDARDIZE
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 6. APPLY SMOTE TO TRAINING DATA
# ---------------------------------------------------------

smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_scaled,
    y_train
)


# ---------------------------------------------------------
# 7. CHECK RESULTS
# ---------------------------------------------------------

print("\nOriginal training shape:")
print(X_train.shape)

print("\nScaled training shape:")
print(X_train_scaled.shape)

print("\nTraining shape after SMOTE:")
print(X_train_balanced.shape)

print("\nTest shape:")
print(X_test_scaled.shape)

print("\nClass distribution after SMOTE:")
print(y_train_balanced.value_counts())


# ---------------------------------------------------------
# 8. CHECK STANDARDIZATION
# ---------------------------------------------------------

print("\nMean of first 5 standardized features:")

print(
    X_train_scaled[:, :5].mean(axis=0)
)

print("\nStandard deviation of first 5 standardized features:")

print(
    X_train_scaled[:, :5].std(axis=0)
)