import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sensor_data.csv")

# Remove target column
features = df.drop(columns=["Pass/Fail"])

# Select first 6 sensor features
selected_features = features.columns[:6]

print("=" * 60)
print("UNIVARIATE ANALYSIS")
print("=" * 60)

print("\nSelected features:")
print(selected_features.tolist())

# ---------------------------------------------------------
# 1. HISTOGRAMS
# ---------------------------------------------------------

for feature in selected_features:

    plt.figure(figsize=(7, 5))

    sns.histplot(
        data=df,
        x=feature,
        kde=True
    )

    plt.title(
        f"Distribution of Feature {feature}"
    )

    plt.xlabel(
        f"Feature {feature}"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/feature_{feature}_distribution.png"
    )

    plt.show()


# ---------------------------------------------------------
# 2. BOXPLOTS
# ---------------------------------------------------------

for feature in selected_features:

    plt.figure(figsize=(7, 4))

    sns.boxplot(
        x=df[feature]
    )

    plt.title(
        f"Boxplot of Feature {feature}"
    )

    plt.xlabel(
        f"Feature {feature}"
    )

    plt.tight_layout()

    plt.savefig(
        f"outputs/feature_{feature}_boxplot.png"
    )

    plt.show()