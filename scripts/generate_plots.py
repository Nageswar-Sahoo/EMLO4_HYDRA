import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Create training loss plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train/loss"], label="Training Loss", color="b")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss over Steps")
plt.legend()
plt.savefig("train_loss.png")
plt.close()

# Create training accuracy plot
plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["train/acc"], label="Training Accuracy", color="g")
plt.xlabel("Step")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Steps")
plt.legend()
plt.savefig("train_acc.png")
plt.close()

# ROC Curve and Confusion Matrix Plotting

# Simulate predictions and labels for testing purposes
# Replace these with your model's actual predictions and labels
# For demonstration, assuming binary classification
y_true = np.random.randint(0, 2, 100)  # Ground truth labels
y_pred = np.random.randint(0, 2, 100)  # Predicted labels
y_scores = np.random.rand(100)  # Prediction scores for ROC

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# Generate test metrics table
test_metrics = df.iloc[-1]
test_table = "| Metric | Value |\n|--------|-------|\n"
test_table += f"| Val Accuracy | {test_metrics['val/acc']:.4f} |\n"
test_table += f"| Val Loss | {test_metrics['val/loss']:.4f} |\n"

# Write the test metrics table to a file
with open("test_metrics.md", "w") as f:
    f.write(test_table)

print("Plots and test metrics table generated successfully.")
