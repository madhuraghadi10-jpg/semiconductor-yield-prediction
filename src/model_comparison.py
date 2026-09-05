import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 1. LOAD MODEL RESULTS
# ---------------------------------------------------------

results = pd.read_csv(
    "outputs/model_comparison.csv"
)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nResults:")
print(results.to_string(index=False))


# ---------------------------------------------------------
# 2. DISPLAY ACCURACY
# ---------------------------------------------------------

print("\nTrain vs Test Accuracy:")

for _, row in results.iterrows():

    print(
        f"{row['Model']}: "
        f"Train = {row['Train Accuracy']:.4f}, "
        f"Test = {row['Test Accuracy']:.4f}"
    )


# ---------------------------------------------------------
# 3. CREATE COMPARISON GRAPH
# ---------------------------------------------------------

models = results["Model"]

train_accuracy = results["Train Accuracy"]

test_accuracy = results["Test Accuracy"]

x = range(len(models))

width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    train_accuracy,
    width=width,
    label="Train Accuracy"
)

plt.bar(
    [i + width / 2 for i in x],
    test_accuracy,
    width=width,
    label="Test Accuracy"
)

plt.xticks(
    list(x),
    models,
    rotation=15
)

plt.ylabel("Accuracy")

plt.xlabel("Machine Learning Model")

plt.title(
    "Comparison of Machine Learning Models"
)

plt.ylim(0, 1)

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png"
)

plt.show()


# ---------------------------------------------------------
# 4. BEST MODEL
# ---------------------------------------------------------

best_index = results[
    "Test Accuracy"
].idxmax()

best_model = results.loc[
    best_index
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Model:",
    best_model["Model"]
)

print(
    "Test Accuracy:",
    best_model["Test Accuracy"]
)

print(
    "ROC-AUC:",
    best_model["ROC-AUC"]
)

print(
    "Best Parameters:",
    best_model["Best Parameters"]
)