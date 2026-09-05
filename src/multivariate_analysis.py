import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sensor_data.csv")

# Separate sensor features
features = df.drop(columns=["Pass/Fail"])

# Find 15 features with highest standard deviation
top_features = (
    features.std()
    .sort_values(ascending=False)
    .head(15)
    .index
)

print("=" * 60)
print("MULTIVARIATE ANALYSIS")
print("=" * 60)

print("\nSelected features:")
print(top_features.tolist())

# Calculate correlation matrix
correlation_matrix = features[
    top_features
].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(2))


# ---------------------------------------------------------
# CORRELATION HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Heatmap of Selected Sensor Features"
)

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png"
)

plt.show()