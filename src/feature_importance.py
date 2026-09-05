import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------
# 1. LOAD CLEANED DATA
# ---------------------------------------------------------

df = pd.read_csv(
    "data/cleaned_sensor_data.csv"
)


# ---------------------------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------

X = df.drop(
    columns=["Pass/Fail"]
)

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
# 4. LOAD BEST MODEL
# ---------------------------------------------------------

model = joblib.load(
    "models/best_model.pkl"
)


# ---------------------------------------------------------
# 5. CALCULATE PERMUTATION IMPORTANCE
# ---------------------------------------------------------

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print("\nCalculating feature importance...")

importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1
)


# ---------------------------------------------------------
# 6. CREATE IMPORTANCE TABLE
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ---------------------------------------------------------
# 7. DISPLAY TOP 20
# ---------------------------------------------------------

print("\nTop 20 important sensor features:")

print(
    importance_df.head(20).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 8. SAVE RESULTS
# ---------------------------------------------------------

importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)


# ---------------------------------------------------------
# 9. VISUALIZE TOP 15
# ---------------------------------------------------------

top_features = importance_df.head(15)

plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Sensor Feature"
)

plt.title(
    "Top 15 Sensor Features"
)

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.show()


print(
    "\nFeature importance analysis completed."
)

print(
    "Results saved to:"
)

print(
    "outputs/feature_importance.csv"
)

print(
    "outputs/feature_importance.png"
)