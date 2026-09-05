import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
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
# 4. DEFINE MODELS
# ---------------------------------------------------------

models = {

    "Logistic Regression": {

        "pipeline": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(
                max_iter=3000,
                random_state=42
            ))
        ]),

        "parameters": {
            "classifier__C": [0.01, 0.1, 1, 10]
        }
    },


    "Random Forest": {

        "pipeline": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "parameters": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5]
        }
    },


    "SVM": {

        "pipeline": ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", SVC(
                probability=True,
                random_state=42
            ))
        ]),

        "parameters": {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": [
                "scale",
                "auto"
            ]
        }
    }
}


# ---------------------------------------------------------
# 5. GRID SEARCH
# ---------------------------------------------------------

results = []

best_model = None
best_model_name = None
best_test_accuracy = 0


for name, model_data in models.items():

    print("\n" + "=" * 60)
    print("GRID SEARCH:", name)
    print("=" * 60)

    grid = GridSearchCV(
        estimator=model_data["pipeline"],
        param_grid=model_data["parameters"],
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(grid.best_params_)

    print("\nBest Cross-Validation Accuracy:")
    print(grid.best_score_)

    # Best model
    model = grid.best_estimator_

    # Predictions
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    # Accuracy
    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    # ROC-AUC
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nTraining Accuracy:")
    print(train_accuracy)

    print("\nTesting Accuracy:")
    print(test_accuracy)

    print("\nROC-AUC:")
    print(roc_auc)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            test_predictions,
            target_names=["Pass", "Fail"]
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            test_predictions
        )
    )

    # Store results
    results.append({
        "Model": name,
        "CV Accuracy": grid.best_score_,
        "Train Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "ROC-AUC": roc_auc,
        "Best Parameters": str(grid.best_params_)
    })

    # Find best model
    if test_accuracy > best_test_accuracy:

        best_test_accuracy = test_accuracy
        best_model = model
        best_model_name = name


# ---------------------------------------------------------
# 6. MODEL COMPARISON
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ---------------------------------------------------------
# 7. SAVE RESULTS
# ---------------------------------------------------------

results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# 8. SAVE BEST MODEL
# ---------------------------------------------------------

import joblib

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print("Model:", best_model_name)
print("Test Accuracy:", best_test_accuracy)

print("\nBest model saved to:")
print("models/best_model.pkl")