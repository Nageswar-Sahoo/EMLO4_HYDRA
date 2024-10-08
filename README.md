<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker </h1>

This repository contains a PyTorch Lightning-based project for classifying dog breeds using a dataset from Kaggle. The project includes Docker support, a DevContainer setup, and inference using a pre-trained model. Configuration management is handled using Hydra.

Features
   
           1> Train and evaluate a deep learning model on the Dog Breed Image Dataset
	   
           2> Docker containerization for easy deployment and reproducibility
	   	   
           3> Hydra for flexible configuration management
	   
           4> eval.py to run evaluation on the validation dataset and print metrics
	   
           5> infer.py for inference on sample images

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

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/train.py

          Above docker script will generate the model artifact in below directory .
    
          <img width="470" alt="image" src="https://github.com/user-attachments/assets/6f518288-0ecd-4995-8eb9-4a73319d6f0f">



 
 5. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/eval.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 
    
 7. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/infer.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 


 9. By default it performs inference on the images present in the input_images folder.

          To modify the infer arguments, you can do the following:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogbreed-classification infer --input_folder="path/to/custom/input" --  output_folder="path/to/custom/output" --     ckpt_path="path/to/custom/checkpoint.ckpt"



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


<img width="1075" alt="image" src="https://github.com/user-attachments/assets/6a844b1a-9553-4c19-8a00-9d6241ace6ec">

<img width="1262" alt="image" src="https://github.com/user-attachments/assets/b56d2f54-71e9-4ce5-9a76-60aef51cfa10">


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


<h3>Requirements</h3>

         Docker
         Kaggle API (for downloading the dataset)
         GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

  
    

