<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker , Hydra , DVC , Optuna </h1>

This repository hosts a PyTorch Lightning project designed for classifying dog breeds, using a dataset from Kaggle. The setup includes Docker support, a DevContainer environment, and inference capabilities with a pre-trained model. Configuration management is powered by Hydra, and hyperparameter optimization is facilitated through Optuna.

Features
   
           1> Train and evaluate a deep learning model on the Dog Cat Image Dataset
	   
           2> Docker containerization for easy deployment and reproducibility
	   	   
           3> Hydra for flexible configuration management
	   
           4> eval.py to run evaluation on the validation dataset and print metrics
	   
           5> infer.py for inference on sample images
	   
           6>  DVC as Data version tool  

           7>  Hyperparameter optimization is facilitated through Optuna



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



# DVC and Google Drive Integration

This project uses **DVC (Data Version Control)** for managing large datasets and model files, with **Google Drive** configured as the remote storage for version control.

## Prerequisites

Before proceeding, ensure you have the following installed:

- [DVC](https://dvc.org/doc/install) (version 2.0+)
- Google Drive account
- [Google Drive API enabled](https://developers.google.com/drive/api/v3/quickstart/python)

## Steps to Set Up DVC with Google Drive

### 1. Initialize DVC in Your Project

    First, initialize DVC in your project if you haven’t already:

    ```bash
    dvc init

### 2. Set Up Google Drive as a DVC Remote
      To configure Google Drive as a DVC remote, follow these steps:

      Create a folder in your Google Drive to serve as the remote storage.

      Run the following command to add Google Drive as your DVC remote:


      dvc remote add -d gdrive_remote gdrive://<folder_id>
      Replace <folder_id> with the Google Drive folder ID. You can get this from the URL of the folder:
      https://drive.google.com/drive/folders/<folder_id>
      Authenticate DVC with Google Drive by running:
      dvc remote modify gdrive_remote gdrive_use_service_account true
### 3. Track Data with DVC
       To track files or directories (e.g., datasets, models), use the dvc add command:

       dvc add <path_to_your_data_or_model>

### 4. Push Data to Google Drive

        To push tracked data to your Google Drive remote, run:

        dvc push

 ### 5. Pull Data from Google Drive
        To pull the latest version of the data from Google Drive, use:

        dvc pull
 ### 6. Version Control with Git
         Make sure to commit and push .dvc files and DVC metadata to your Git repository:
	 git add <tracked_files.dvc> .dvc/config .dvcignore
	 git commit -m "Track data with DVC"
	 git push

<h2>Hyperparameter Tuning with Optuna</h2>

Hyperparameter Tuning with Optuna
Optuna is an efficient and easy-to-use hyperparameter optimization library designed to automate the process of finding the best parameters for your model. By leveraging Optuna, you can maximize your model's performance through an intelligent search process over hyperparameter space.

<h3>Installation</h3>
To use Optuna, first ensure it’s installed in your environment:
    pip install optuna
    

<h4>Below is the Optuna hyper param with Hydra config : </h4>

          hydra:
            sweeper:
              sampler:
                _target_: optuna.samplers.TPESampler
               seed: 123
               n_startup_trials: 3
               direction: maximize
               study_name: catdog_vit_hparam_optimization
               storage: null
               n_trials: 10
               n_jobs: 1

         params:
            # Model architecture params
            model.drop_rate: interval(0.0, 0.3)
            model.drop_path_rate: interval(0.0, 0.3)
            model.head_init_scale: interval(0.5, 2.0)
            data.batch_size: choice(32, 64, 128, 256)

<h4>RUN : </h4>  python src/train.py --multirun hydra/launcher=joblib hparam=catdog_vit_hparam +trainer.log_every_n_steps=5 hydra.sweeper.n_jobs=4

<h1>Hydra-Optuna Sweeper Configuration</h1>

Above configuration file integrates Hydra and Optuna to manage hyperparameter optimization for a model using the Tree-structured Parzen Estimator (TPE) sampler. The settings are designed to maximize the model's performance.

<h4>Sweeper: </h4> Configures Hydra to use Optuna for sweeping hyperparameters.</h1>

<h4>Sampler: </h4> Defines the Optuna TPESampler as the optimization algorithm, with a fixed random seed for reproducibility and n_startup_trials set to 3 for initial random explorations.</h1>

<h4>Study: </h4> The study_name is set to catdog_vit_hparam_optimization, enabling tracking for this specific experiment. The direction is set to maximize, meaning we aim to maximize the objective function (e.g., accuracy).

<h4>Trials and Jobs: </h4> The n_trials parameter is set to 10, specifying the number of trials to run, and n_jobs is set to 1, so each trial runs sequentially.
<h4>Hyperparameters (params):</h4>

   <h4>Model architecture: </h4> Hyperparameters for model regularization, such as drop_rate and drop_path_rate, are explored within a range of 0.0 to 0.3. The scaling factor, head_init_scale, is varied between 0.5 and 2.0.</h4>

   <h4>Batch Size: </h4> The batch size (data.batch_size) is chosen from discrete options (32, 64, 128, 256), allowing exploration of different data processing loads.

This setup offers a balance between exploration and computational efficiency, enabling a quick sweep over a key set of hyperparameters for initial performance tuning.
      
              


<h2>Using Hydra for Configuration Management</h2>
The project utilizes Hydra to manage configurations. Configuration files are located in the configs/ directory. You can modify these files to adjust various parameters for training, evaluation, and inference.


Running with Hydra
To run the training script with a specified configuration, use the following command:

        python src/train.py 
	    python src/eval.py 
        python src/infer.py 

All Hydra configurations are located in the following directory. Any updates to the parameters can be made there.

<img width="663" alt="image" src="https://github.com/user-attachments/assets/dac805bd-d788-4907-b710-589d12f3ee20">




### Train and Report DVC Pipeline

This project uses a GitHub Actions pipeline to automate the process of training machine learning models and generating reports using DVC and CML (Continuous Machine Learning). The pipeline is triggered on any push or pull request to the origin/feature/dvc or main branches.

The pipeline performs the following key steps:

             1. Checkout Code: Pulls the latest changes from the repository.
	     
             2. Setup Environment: Installs necessary dependencies, including Python 3.12 and DVC, and sets up the environment.
	     
             3. Pull Data from DVC: Retrieves the dataset from DVC storage (Google Drive in this case).
	     
             4. Train Model: Runs the  dvc repro command to train the model based on the DVC pipeline.
	     
             5. Generate CML Report: Produces a report that includes training metrics (e.g., accuracy, loss) and test results, which is automatically posted as a comment in the relevant pull request.
	     
             6. This automated workflow ensures that model training and reporting are continuously integrated into the development process, improving collaboration and transparency across team members.

 ### How to Train, Evaluate, and Infer Using Docker

  ### 1. Build the Docker image:

          docker build -t dogcat-classification .
 
  ### 2. To run training:

          docker run -v $(pwd)/model_artifacts:/app/logs dogcat-classification src/train.py

          Above docker script will generate the model artifact in below directory .
    
 <img width="470" alt="image" src="https://github.com/user-attachments/assets/6f518288-0ecd-4995-8eb9-4a73319d6f0f">
 
 ###  3. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/logs dogcat-classification src/eval.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 
    
 ###  4. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/logs dogcat-classification src/infer.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 


 ###  5 By default it performs inference on the images present in the dataset folder.

          To modify the infer arguments, you can do the following:

          docker run -v $(pwd)/model_artifacts:/app/checkpoints dogcat-classification infer --input_folder="path/to/custom/input" --  output_folder="path/to/custom/output" --     ckpt_path="path/to/custom/checkpoint.ckpt"




 ### <h3>Docker Setup</h3>

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
 ###  Model  Summery : 

ConvNeXt is a modernized version of the classic convolutional neural network (CNN) architecture that was designed to compete with Vision Transformers (ViTs) in terms of performance on image classification and other vision tasks. Created by Facebook AI, it was inspired by ResNet and modernized with architectural choices commonly found in transformers
<img width="627" alt="image" src="https://github.com/user-attachments/assets/a1f60412-c4d6-4c6c-92a4-5c827d5f1508">


 
      


ouput from :  src/train.py 
<img width="1320" alt="image" src="https://github.com/user-attachments/assets/69f341bf-4922-4ea6-bb8e-cd1d272ba623">
ouput from :  src/infer.py
<img width="1784" alt="image" src="https://github.com/user-attachments/assets/c32b5517-337b-4655-a3a8-24e739019222">



<h3>Model Training Results </h3>

  The model prediction gets saved in the predicted_images folder in the model artifacts.


  <table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ea8181c3-e3d2-4fed-9a4e-b24123b7fabc" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/9cb2c6dd-7631-46f4-bb13-f78f72e0c20e" width="500"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/ed4e8bf3-0cb9-4927-818d-658d1e5e05e1" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/9929a2c7-d860-486a-a62d-80783b4fda17" width="500"/></td>
  </tr>
   <tr>
    <td><img src="https://github.com/user-attachments/assets/8627575f-148e-4af8-93f8-036972ee503f" width="500"/></td>
    <td><img src="https://github.com/user-attachments/assets/192cd16a-85c3-4a50-88e5-b644ef2c5854" width="500"/></td>
  </tr>
</table>

![image](https://github.com/user-attachments/assets/fd741a76-a17a-46ab-9d44-3696cfdbf5d1)



<h3>Requirements</h3>

         Docker
         Kaggle API (for downloading the dataset)
         GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

