import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Print DataFrame summary
print("DataFrame head:")
print(df.head())  # Print first 5 rows

print("\nDataFrame info:")
df.info()

print("\nDataFrame description:")
print(df.describe())

# Filter rows where train/acc or train/loss are not NaN for plotting
train_metrics = df.dropna(subset=["train/acc", "train/loss"])

# Create training loss plot if there are valid rows
if not train_metrics.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics["step"], train_metrics["train/loss"], label="Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss over Steps")
    plt.legend()
    plt.savefig("train_loss.png")
    plt.close()

# Create training accuracy plot if there are valid rows
if not train_metrics.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics["step"], train_metrics["train/acc"], label="Training Accuracy")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy over Steps")
    plt.legend()
    plt.savefig("train_acc.png")
    plt.close()

# Handle validation metrics: only include rows where val/acc and val/loss are not NaN
valid_val_metrics = df.dropna(subset=["val/acc", "val/loss"])

# Create validation loss plot if validation metrics exist
if not valid_val_metrics.empty:
    plt.figure(figsize=(10, 6))
    plt.plot(valid_val_metrics["step"], valid_val_metrics["val/loss"], label="Validation Loss", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss over Steps")
    plt.legend()
    plt.savefig("val_loss.png")
    plt.close()

    # Create validation accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(valid_val_metrics["step"], valid_val_metrics["val/acc"], label="Validation Accuracy", color="green")
    plt.xlabel("Step")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy over Steps")
    plt.legend()
    plt.savefig("val_acc.png")
    plt.close()

# Generate test metrics table (use the last available validation metrics)
if not valid_val_metrics.empty:
    test_metrics = valid_val_metrics.iloc[-1]
    test_table = "| Metric      | Value      |\n|-------------|------------|\n"
    test_table += f"| Val Accuracy | {test_metrics['val/acc']:.4f} |\n"
    test_table += f"| Val Loss     | {test_metrics['val/loss']:.4f} |\n"

    # Write the test metrics table to a file
    with open("test_metrics.md", "w") as f:
        f.write(test_table)
else:
    print("No validation metrics found, skipping test metrics table generation.")