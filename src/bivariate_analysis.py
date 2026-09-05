import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sensor_data.csv")

# Convert target to 0/1 for correlation analysis
df["Target"] = df["Pass/Fail"].map({
    -1: 0,
    1: 1
})

# Get sensor features only
features = df.drop(columns=["Pass/Fail", "Target"])

# Calculate correlation with target
correlations = features.corrwith(df["Target"])

# Sort by absolute correlation
top_features = (
    correlations.abs()
    .sort_values(ascending=False)
    .head(6)
    .index
)

print("=" * 60)
print("BIVARIATE ANALYSIS")
print("=" * 60)

print("\nTop 6 features related to Pass/Fail:")
print(top_features.tolist())

print("\nCorrelation with target:")

for feature in top_features:
    print(
        f"{feature}: {correlations[feature]:.4f}"
    )


# ---------------------------------------------------------
# BOXPLOTS: FEATURE vs PASS/FAIL
# ---------------------------------------------------------

for feature in top_features:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Pass/Fail",
        y=feature
    )

    plt.title(
        f"Feature {feature} vs Pass/Fail"
    )

    plt.xlabel("Yield Result")
    plt.ylabel(f"Feature {feature}")

    plt.tight_layout()

    plt.savefig(
        f"outputs/feature_{feature}_vs_target.png"
    )

    plt.show()