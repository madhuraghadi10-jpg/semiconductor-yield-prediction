import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------

df = pd.read_csv("data/cleaned_sensor_data.csv")

print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ---------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(columns=["Pass/Fail"])
y = df["Pass/Fail"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)


# ---------------------------------------------------------
# 3. TARGET DISTRIBUTION
# ---------------------------------------------------------

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(
    y.value_counts(normalize=True) * 100
)


# ---------------------------------------------------------
# 4. CONVERT TARGET
# ---------------------------------------------------------

# Pass (-1) → 0
# Fail (1)  → 1

y = y.map({
    -1: 0,
    1: 1
})

print("\nConverted target distribution:")
print(y.value_counts())


# ---------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ---------------------------------------------------------
# 6. CHECK DATASET CHARACTERISTICS
# ---------------------------------------------------------

print("\nOriginal target percentage:")
print(
    y.value_counts(normalize=True) * 100
)

print("\nTraining target percentage:")
print(
    y_train.value_counts(normalize=True) * 100
)

print("\nTesting target percentage:")
print(
    y_test.value_counts(normalize=True) * 100
)