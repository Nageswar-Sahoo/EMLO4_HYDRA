<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker </h1>

This repository contains a PyTorch Lightning-based project for classifying dog breeds using a dataset from Kaggle. The project includes Docker support, a DevContainer setup, and inference using a pre-trained model. Configuration management is handled using Hydra.

Features
   
   Train and evaluate a deep learning model on the Dog Breed Image Dataset
   Docker containerization for easy deployment and reproducibility
   DevContainer setup for development
   Hydra for flexible configuration management
   eval.py to run evaluation on the validation dataset and print metrics
   infer.py for inference on sample images

<h3>About Dataset</h3>

Description

This dataset contains a collection of images for 10 different dog breeds, meticulously gathered and organized to facilitate various computer vision tasks such as image classification and object detection. The dataset includes the following breeds:

       Golden Retriever
       German Shepherd
       Labrador Retriever
       Bulldog
       Beagle
       Poodle
       Rottweiler
       Yorkshire Terrier
       Boxer
       Dachshund

Each breed is represented by 100 images, stored in separate directories named after the respective breed. The images have been curated to ensure diversity and relevance, making this dataset a valuable resource for training and evaluating machine learning models in the field of computer vision.

<h2>Using Hydra for Configuration Management</h2>
The project utilizes Hydra to manage configurations. Configuration files are located in the configs/ directory. You can modify these files to adjust various parameters for training, evaluation, and inference.

Running with Hydra
To run the training script with a specified configuration, use the following command:

        python src/train.py 
	python src/eval.py 
        python src/infer.py 



Model 
    Model Architecture

How to Train, Evaluate, and Infer Using Docker

 1. Build the Docker image:

          docker build -t dogbreed-classification .
 
 3. To run training:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification train
 
 5. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification eval
    
 7. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer

 8. By default it performs inference on the images present in the input_images folder.

          To modify the infer arguments, you can do the following:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer --input_folder="path/to/custom/input" --  output_folder="path/to/custom/output" --     ckpt_path="path/to/custom/checkpoint.ckpt"


<h2>Scripts Overview</h2>
1. train.py
This script handles training the model using a DataModule for the dataset. It saves the best checkpoint during training.

2. eval.py
This script loads the model from a checkpoint and evaluates it on the validation dataset.

3. infer.py
This script runs inference on a folder of images and saves the predictions.

<h3>Prediction Results</h3>

  The model prediction gets saved in the predicted_images folder in the model artifacts.


  <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/92cb4dd8-8f37-4ae5-a906-7447554720a9" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/2cde3ce2-3596-4bf9-bb88-9d2c996407eb" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c2a4bc37-078d-49d2-afbe-65cc0ccd3a82" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/7ca0ca0e-3295-49eb-a266-e2009eca271e" width="500"/></td>
  </tr>
   <tr>
    <td><img src="https://github.com/user-attachments/assets/97d12ed1-9f40-4d9e-929f-d282be82e882" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/3adf2fc0-2753-4cb9-bd93-b4bbcb236077" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c6bb007a-d7be-4942-8b04-2d8148e57522" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/2d96e61d-b49b-41cb-b342-871f2a43476d" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/20e0d992-23dc-4816-a6d1-d01efd04b0bb" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/c5ebc356-60f7-4ab1-bcca-22cfc5e281de" width="500"/></td>
  </tr>
</table>

<h3>Docker Setup</h3>

1. Dockerfile
This repository includes a Dockerfile to containerize the training, evaluation, and inference process. The Docker image includes the necessary dependencies and installs the project package.
         # Build stage
         FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

         ENV UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
         ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

         WORKDIR /app

         # Install dependencies
         RUN --mount=type=cache,target=/root/.cache/uv \
	--mount=type=bind,source=uv.lock,target=uv.lock \
	--mount=type=bind,source=pyproject.toml,target=pyproject.toml \
	uv sync --frozen --no-install-project --no-dev

         # Copy the rest of the application
         ADD . /app

         # Install the project and its dependencies
         RUN --mount=type=cache,target=/root/.cache/uv \
	uv sync --frozen --no-dev

         # Final stage
         FROM python:3.12-slim-bookworm

         # Copy the application from the builder
         COPY --from=builder --chown=app:app /app /app

         # Place executables in the environment at the front of the path
         ENV PATH="/app/.venv/bin:$PATH"

         # Set the working directory
         WORKDIR /app


<h3>Requirements</h3>
Docker
Kaggle API (for downloading the dataset)
GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

  
    

