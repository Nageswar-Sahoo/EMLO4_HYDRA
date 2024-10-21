import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import numpy as np


# Find the most recent metrics.csv file
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

print(df)

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

# # Function to plot confusion matrix
# def plot_confusion_matrix(csv_path, title="Confusion Matrix", output_image_path="confusion_matrix.png"):
#     # Load confusion matrix from the CSV file
#     cm_df = pd.read_csv(csv_path)

#     # Plot confusion matrix heatmap
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.title(title)

#     # Save the confusion matrix image
#     plt.savefig(output_image_path)
#     plt.close()
#     print(f"Confusion matrix image saved to {output_image_path}")

# train_confusion_csv_path = "logs/train_confusion_matrix_details.csv"
# val_confusion_csv_path = "logs/val_confusion_matrix_details.csv"

# # Check if the train confusion matrix CSV exists
# if os.path.exists(train_confusion_csv_path):
#     plot_confusion_matrix(train_confusion_csv_path, title="Training Confusion Matrix", output_image_path="train_confusion_matrix.png")
# else:
#     print(f"No training confusion matrix CSV found at {train_confusion_csv_path}")

# # Check if the validation confusion matrix CSV exists
# if os.path.exists(val_confusion_csv_path):
#     plot_confusion_matrix(val_confusion_csv_path, title="Validation Confusion Matrix", output_image_path="val_confusion_matrix.png")
# else:
#     print(f"No validation confusion matrix CSV found at {val_confusion_csv_path}")

# # Generate test metrics table (use the last available validation metrics)
# if not valid_val_metrics.empty:
#     test_metrics = valid_val_metrics.iloc[-1]
#     test_table = "| Metric      | Value      |\n|-------------|------------|\n"
#     test_table += f"| Val Accuracy | {test_metrics['val/acc']:.4f} |\n"
#     test_table += f"| Val Loss     | {test_metrics['val/loss']:.4f} |\n"

#     # Add confusion matrix references in the table
#     if os.path.exists("train_confusion_matrix.png"):
#         test_table += "| Train Confusion Matrix | ![Train Confusion Matrix](train_confusion_matrix.png) |\n"
#     if os.path.exists("val_confusion_matrix.png"):
#         test_table += "| Val Confusion Matrix | ![Val Confusion Matrix](val_confusion_matrix.png) |\n"

#     # Write the test metrics table to a file
#     with open("test_metrics.md", "w") as f:
#         f.write(test_table)
# else:
#     print("No validation metrics found, skipping test metrics table generation.")


# Function to plot confusion matrix
def plot_confusion_matrix(csv_path, title="Confusion Matrix", output_image_path="confusion_matrix.png"):
    # Load confusion matrix from the CSV file
    cm_df = pd.read_csv(csv_path)

    # Plot confusion matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)

    # Save the confusion matrix image
    plt.savefig(output_image_path)
    plt.close()
    print(f"Confusion matrix image saved to {output_image_path}")

def plot_predicted_images(num_images=16, output_file="predicted_images.png"):
    # Get all image files from the logs/infer/runs folders
    image_files = glob("logs/infer/runs/*/*.png")
    
    if not image_files:
        print("No predicted images found.")
        return None
    
    # If there are fewer than num_images, use all available images
    num_images = min(num_images, len(image_files))
    
    # Randomly select num_images from the folder
    selected_images = random.sample(image_files, num_images)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_images)))
    
    # Create a grid of subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    fig.suptitle("Sample Predicted Images", fontsize=16)
    
    # Plot each image
    for i, image_path in enumerate(selected_images):
        img = plt.imread(image_path)
        row = i // grid_size
        col = i % grid_size
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {i+1}")
    
    # Remove any empty subplots
    for i in range(num_images, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        fig.delaxes(axes[row, col])
    
    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Predicted images plot saved to {output_file}")
    return output_file
    
# Get all confusion matrix CSV files for training and validation
train_confusion_csv_files = glob("logs/train_confusion_matrix_epoch_*.csv")
val_confusion_csv_files = glob("logs/val_confusion_matrix_epoch_*.csv")

# Plot and save all confusion matrices for each epoch in training
if train_confusion_csv_files:
    for csv_file in train_confusion_csv_files:
        epoch = os.path.basename(csv_file).split("_")[-1].split(".")[0]  # Extract epoch number from filename
        plot_confusion_matrix(csv_file, 
                              title=f"Training Confusion Matrix - Epoch {epoch}", 
                              output_image_path=f"train_confusion_matrix_epoch_{epoch}.png")
else:
    print("No training confusion matrix CSV files found.")

# Plot and save all confusion matrices for each epoch in validation
if val_confusion_csv_files:
    for csv_file in val_confusion_csv_files:
        epoch = os.path.basename(csv_file).split("_")[-1].split(".")[0]  # Extract epoch number from filename
        plot_confusion_matrix(csv_file, 
                              title=f"Validation Confusion Matrix - Epoch {epoch}", 
                              output_image_path=f"val_confusion_matrix_epoch_{epoch}.png")
else:
    print("No validation confusion matrix CSV files found.")

# Generate test metrics table (use the last available validation metrics)
csv_files = glob("logs/train/runs/*/csv/version_*/metrics.csv")
if not csv_files:
    raise FileNotFoundError("No metrics.csv file found")
latest_csv = max(csv_files, key=os.path.getctime)

# Read the CSV file
df = pd.read_csv(latest_csv)

# Handle validation metrics: only include rows where val/acc and val/loss are not NaN
valid_val_metrics = df.dropna(subset=["val/acc", "val/loss"])

# Get all confusion matrix image paths for train and validation
train_confusion_images = sorted(glob("train_confusion_matrix_epoch_*.png"))
val_confusion_images = sorted(glob("val_confusion_matrix_epoch_*.png"))

# Generate test metrics table including confusion matrix images from the last epoch
if not valid_val_metrics.empty:
    test_metrics = valid_val_metrics.iloc[-1]
    test_table = "| Metric | Value |\n|--------|-------|\n"
    test_table += f"| Val Accuracy | {test_metrics['val/acc']:.4f} |\n"
    test_table += f"| Val Loss | {test_metrics['val/loss']:.4f} |\n"

    # Add all confusion matrix image references to the test table
    for image in train_confusion_images:
        epoch = image.split('_')[-1].split('.')[0]  # Extract epoch number from filename
        test_table += f"| Train Confusion Matrix (Epoch {epoch}) | ![Train Confusion Matrix Epoch {epoch}]({image}) |\n"

    for image in val_confusion_images:
        epoch = image.split('_')[-1].split('.')[0]  # Extract epoch number from filename
        test_table += f"| Val Confusion Matrix (Epoch {epoch}) | ![Val Confusion Matrix Epoch {epoch}]({image}) |\n"

    # Add predicted images to the test table
    predicted_images_file = plot_predicted_images(num_images=16, output_file="predicted_images.png")
    if predicted_images_file:
        test_table += f"| Predicted Images | ![Predicted Images]({predicted_images_file}) |\n"

    # Write the test metrics table to a file
    with open("test_metrics.md", "w") as f:
        f.write(test_table)
else:
    print("No validation metrics found, skipping test metrics table generation.")
    


