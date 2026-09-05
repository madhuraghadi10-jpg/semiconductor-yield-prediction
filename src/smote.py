import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------

df = pd.read_csv("data/cleaned_sensor_data.csv")

print("=" * 60)
print("SMOTE - CLASS BALANCING")
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
# 4. BEFORE SMOTE
# ---------------------------------------------------------

print("\nClass distribution BEFORE SMOTE:")

print(y_train.value_counts())

print("\nPercentage BEFORE SMOTE:")

print(
    y_train.value_counts(normalize=True) * 100
)


# ---------------------------------------------------------
# 5. APPLY SMOTE
# ---------------------------------------------------------

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# ---------------------------------------------------------
# 6. AFTER SMOTE
# ---------------------------------------------------------

print("\nClass distribution AFTER SMOTE:")

print(y_train_smote.value_counts())

print("\nPercentage AFTER SMOTE:")

print(
    y_train_smote.value_counts(normalize=True) * 100
)


# ---------------------------------------------------------
# 7. SHAPES
# ---------------------------------------------------------

print("\nOriginal training data:")
print(X_train.shape)

print("\nTraining data after SMOTE:")
print(X_train_smote.shape)

print("\nTest data:")
print(X_test.shape)