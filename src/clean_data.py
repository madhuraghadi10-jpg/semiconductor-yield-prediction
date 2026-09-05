import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("data/sensor-data.csv")

print("Original dataset shape:", df.shape)


# ---------------------------------------------------------
# 2. REMOVE TIMESTAMP
# ---------------------------------------------------------

if "Time" in df.columns:
    df = df.drop(columns=["Time"])

print("After removing Time:", df.shape)


# ---------------------------------------------------------
# 3. REMOVE FEATURES WITH >50% MISSING VALUES
# ---------------------------------------------------------

target = "Pass/Fail"

X = df.drop(columns=[target])

missing_percentage = X.isnull().sum() / len(X) * 100

columns_to_drop = missing_percentage[
    missing_percentage > 50
].index.tolist()

print("\nFeatures removed because of >50% missing values:")
print(columns_to_drop)

df = df.drop(columns=columns_to_drop)

print(
    "\nNumber of features removed:",
    len(columns_to_drop)
)

print(
    "Shape after removing high-missing features:",
    df.shape
)


# ---------------------------------------------------------
# 4. HANDLE REMAINING MISSING VALUES
# ---------------------------------------------------------

feature_columns = df.drop(
    columns=[target]
).columns

for column in feature_columns:

    if df[column].isnull().sum() > 0:

        median_value = df[column].median()

        df[column] = df[column].fillna(
            median_value
        )


# ---------------------------------------------------------
# 5. CHECK MISSING VALUES
# ---------------------------------------------------------

remaining_missing = df.isnull().sum().sum()

print(
    "\nRemaining missing values:",
    remaining_missing
)


# ---------------------------------------------------------
# 6. SAVE CLEANED DATA
# ---------------------------------------------------------

df.to_csv(
    "data/cleaned_sensor_data.csv",
    index=False
)

print(
    "\nCleaned dataset saved successfully!"
)

print(
    "Final dataset shape:",
    df.shape
)