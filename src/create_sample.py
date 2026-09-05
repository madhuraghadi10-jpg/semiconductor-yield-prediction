import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "data/cleaned_sensor_data.csv"
)

# Remove target column
sample = df.drop(
    columns=["Pass/Fail"]
)

# Take first 5 samples
sample = sample.head(5)

# Save sample input file
sample.to_csv(
    "data/sample_input.csv",
    index=False
)

print("=" * 60)
print("SAMPLE INPUT CREATED")
print("=" * 60)

print("\nShape:")
print(sample.shape)

print("\nSaved to:")
print("data/sample_input.csv")