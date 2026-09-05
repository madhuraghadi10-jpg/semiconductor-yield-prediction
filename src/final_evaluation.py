import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


# ---------------------------------------------------------
# 1. LOAD DATA
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
# 5. PREDICTIONS
# ---------------------------------------------------------

train_predictions = model.predict(
    X_train
)

test_predictions = model.predict(
    X_test
)


# ---------------------------------------------------------
# 6. ACCURACY
# ---------------------------------------------------------

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

print("\nTraining Accuracy:")
print(train_accuracy)

print("\nTesting Accuracy:")
print(test_accuracy)


# ---------------------------------------------------------
# 7. CLASSIFICATION REPORT
# ---------------------------------------------------------

report = classification_report(
    y_test,
    test_predictions,
    target_names=[
        "Pass",
        "Fail"
    ]
)

print("\nClassification Report:")
print(report)


# Save report
with open(
    "outputs/classification_report.txt",
    "w"
) as file:

    file.write(report)


# ---------------------------------------------------------
# 8. CONFUSION MATRIX
# ---------------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Pass",
        "Fail"
    ]
)

display.plot()

plt.title(
    "Final Model Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png"
)

plt.show()


# ---------------------------------------------------------
# 9. ROC-AUC
# ---------------------------------------------------------

if hasattr(
    model,
    "predict_proba"
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nROC-AUC:")
    print(roc_auc)