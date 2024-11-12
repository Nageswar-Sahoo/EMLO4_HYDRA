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


<h3>Setting Up S3 as a DVC Remote for Data Storage</h3>

This guide outlines how to configure an Amazon S3 bucket as a remote storage for DVC (Data Version Control). This setup enables version-controlled data storage in S3, perfect for large datasets in machine learning and data science workflows.

<h4>Prerequisites</h4>

    1.AWS account with access to create and manage S3 buckets.
    
    2.IAM credentials with permissions to read, write, and list objects in the S3 bucket.
    
    3.DVC installed locally (pip install dvc[s3]).
    
    4.Git and GitHub repository setup.

<h4>Steps to Configure S3 as a DVC Remote</h4>

1. Create an S3 Bucket

        Go to the AWS Management Console and create a new S3 bucket (e.g., my-dvc-bucket).

        Note the bucket name and region (e.g., ap-south-1 for Mumbai).

2. Configure IAM Permissions

      Assign permissions to your IAM user to access the S3 bucket. Use the following policy, replacing my-dvc-bucket with your actual bucket name:


        {
          "Version": "2012-10-17",
          "Statement": [
            {
              "Effect": "Allow",
              "Action": [
                "s3:ListBucket"
              ],
              "Resource": "arn:aws:s3:::my-dvc-bucket"
            },
            {
              "Effect": "Allow",
              "Action": [
                "s3:GetObject",
                "s3:PutObject"              ],
              "Resource": "arn:aws:s3:::my-dvc-bucket/*"
            }
          ]
        }

3. Set Up DVC Remote

       In your local Git repository, configure the S3 bucket as the DVC remote:

       dvc remote add -d myremote s3://my-dvc-bucket/path/to/data

4. Push Data to S3 with DVC

        After configuring the remote, add files to DVC, then push to the S3 bucket:

         dvc add data
         git add data .gitignore
         git commit -m "Add large file with DVC"
         dvc push

6. Setting Up GitHub Actions for Automated DVC Pull

       To automate data pulls from S3 in GitHub Actions, add the following workflow file (.github/workflows/dvc_pull.yml):

       name: DVC with S3
   
         steps:
           - name: Checkout Repository
             uses: actions/checkout@v3

           - name: Install DVC and Boto3
             run: |
               pip install dvc[s3] boto3

           - name: Configure AWS Credentials
             run: |
               aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
               aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
               aws configure set region ${{ secrets.AWS_REGION }}

           - name: Set DVC Remote
             run: |
               dvc remote add -d myremote s3://dvc-nagsh-demo
           - name: Enable Debug Logging
             run: export DVC_LOGLEVEL=DEBUG
        
           - name: Pull DVC Data
             run: |
               dvc pull -v
             env:
               AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
               AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
               AWS_REGION: ${{ secrets.AWS_REGION }}

           - name: Verify DVC Data Files
             run: |
               dvc status
     



<h2>Using Hydra for Configuration Management</h2>
The project utilizes Hydra to manage configurations. Configuration files are located in the configs/ directory. You can modify these files to adjust various parameters for training, evaluation, and inference.

Running with Hydra
To run the training script with a specified configuration, use the following command:

        python src/train.py 
	    python src/eval.py 
        python src/infer.py 

All Hydra configurations are located in the following directory. Any updates to the parameters can be made there.

<img width="886" alt="image" src="https://github.com/user-attachments/assets/4150a98c-7f47-44b5-a388-fc11f2ac831a">

 



Model  Summery : 
   
        ┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
        ┃    ┃ Name                        ┃ Type                 ┃ Params ┃ Mode  ┃
        ┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
        │ 0  │ model                       │ ResNet               │ 11.2 M │ train │
        │ 1  │ model.conv1                 │ Conv2d               │  9.4 K │ train │
        │ 2  │ model.bn1                   │ BatchNorm2d          │    128 │ train │
        │ 3  │ model.act1                  │ ReLU                 │      0 │ train │
        │ 4  │ model.maxpool               │ MaxPool2d            │      0 │ train │
        │ 5  │ model.layer1                │ Sequential           │  147 K │ train │
        │ 6  │ model.layer1.0              │ BasicBlock           │ 74.0 K │ train │
        │ 7  │ model.layer1.0.conv1        │ Conv2d               │ 36.9 K │ train │
        │ 8  │ model.layer1.0.bn1          │ BatchNorm2d          │    128 │ train │
        │ 9  │ model.layer1.0.drop_block   │ Identity             │      0 │ train │
        │ 10 │ model.layer1.0.act1         │ ReLU                 │      0 │ train │
        │ 11 │ model.layer1.0.aa           │ Identity             │      0 │ train │
        │ 12 │ model.layer1.0.conv2        │ Conv2d               │ 36.9 K │ train │
        │ 13 │ model.layer1.0.bn2          │ BatchNorm2d          │    128 │ train │
        │ 14 │ model.layer1.0.act2         │ ReLU                 │      0 │ train │
        │ 15 │ model.layer1.1              │ BasicBlock           │ 74.0 K │ train │
        │ 16 │ model.layer1.1.conv1        │ Conv2d               │ 36.9 K │ train │
        │ 17 │ model.layer1.1.bn1          │ BatchNorm2d          │    128 │ train │
        │ 18 │ model.layer1.1.drop_block   │ Identity             │      0 │ train │
        │ 19 │ model.layer1.1.act1         │ ReLU                 │      0 │ train │
        │ 20 │ model.layer1.1.aa           │ Identity             │      0 │ train │
        │ 21 │ model.layer1.1.conv2        │ Conv2d               │ 36.9 K │ train │
        │ 22 │ model.layer1.1.bn2          │ BatchNorm2d          │    128 │ train │
        │ 23 │ model.layer1.1.act2         │ ReLU                 │      0 │ train │
        │ 24 │ model.layer2                │ Sequential           │  525 K │ train │
        │ 25 │ model.layer2.0              │ BasicBlock           │  230 K │ train │
        │ 26 │ model.layer2.0.conv1        │ Conv2d               │ 73.7 K │ train │
        │ 27 │ model.layer2.0.bn1          │ BatchNorm2d          │    256 │ train │
        │ 28 │ model.layer2.0.drop_block   │ Identity             │      0 │ train │
        │ 29 │ model.layer2.0.act1         │ ReLU                 │      0 │ train │
        │ 30 │ model.layer2.0.aa           │ Identity             │      0 │ train │
        │ 31 │ model.layer2.0.conv2        │ Conv2d               │  147 K │ train │
        │ 32 │ model.layer2.0.bn2          │ BatchNorm2d          │    256 │ train │
        │ 33 │ model.layer2.0.act2         │ ReLU                 │      0 │ train │
        │ 34 │ model.layer2.0.downsample   │ Sequential           │  8.4 K │ train │
        │ 35 │ model.layer2.0.downsample.0 │ Conv2d               │  8.2 K │ train │
        │ 36 │ model.layer2.0.downsample.1 │ BatchNorm2d          │    256 │ train │
        │ 37 │ model.layer2.1              │ BasicBlock           │  295 K │ train │
        │ 38 │ model.layer2.1.conv1        │ Conv2d               │  147 K │ train │
        │ 39 │ model.layer2.1.bn1          │ BatchNorm2d          │    256 │ train │
        │ 40 │ model.layer2.1.drop_block   │ Identity             │      0 │ train │
        │ 41 │ model.layer2.1.act1         │ ReLU                 │      0 │ train │
        │ 42 │ model.layer2.1.aa           │ Identity             │      0 │ train │
        │ 43 │ model.layer2.1.conv2        │ Conv2d               │  147 K │ train │
        │ 44 │ model.layer2.1.bn2          │ BatchNorm2d          │    256 │ train │
        │ 45 │ model.layer2.1.act2         │ ReLU                 │      0 │ train │
        │ 46 │ model.layer3                │ Sequential           │  2.1 M │ train │
        │ 47 │ model.layer3.0              │ BasicBlock           │  919 K │ train │
        │ 48 │ model.layer3.0.conv1        │ Conv2d               │  294 K │ train │
        │ 49 │ model.layer3.0.bn1          │ BatchNorm2d          │    512 │ train │
        │ 50 │ model.layer3.0.drop_block   │ Identity             │      0 │ train │
        │ 51 │ model.layer3.0.act1         │ ReLU                 │      0 │ train │
        │ 52 │ model.layer3.0.aa           │ Identity             │      0 │ train │
        │ 53 │ model.layer3.0.conv2        │ Conv2d               │  589 K │ train │
        │ 54 │ model.layer3.0.bn2          │ BatchNorm2d          │    512 │ train │
        │ 55 │ model.layer3.0.act2         │ ReLU                 │      0 │ train │
        │ 56 │ model.layer3.0.downsample   │ Sequential           │ 33.3 K │ train │
        │ 57 │ model.layer3.0.downsample.0 │ Conv2d               │ 32.8 K │ train │
        │ 58 │ model.layer3.0.downsample.1 │ BatchNorm2d          │    512 │ train │
        │ 59 │ model.layer3.1              │ BasicBlock           │  1.2 M │ train │
        │ 60 │ model.layer3.1.conv1        │ Conv2d               │  589 K │ train │
        │ 61 │ model.layer3.1.bn1          │ BatchNorm2d          │    512 │ train │
        │ 62 │ model.layer3.1.drop_block   │ Identity             │      0 │ train │
        │ 63 │ model.layer3.1.act1         │ ReLU                 │      0 │ train │
        │ 64 │ model.layer3.1.aa           │ Identity             │      0 │ train │
        │ 65 │ model.layer3.1.conv2        │ Conv2d               │  589 K │ train │
        │ 66 │ model.layer3.1.bn2          │ BatchNorm2d          │    512 │ train │
        │ 67 │ model.layer3.1.act2         │ ReLU                 │      0 │ train │
        │ 68 │ model.layer4                │ Sequential           │  8.4 M │ train │
        │ 69 │ model.layer4.0              │ BasicBlock           │  3.7 M │ train │
        │ 70 │ model.layer4.0.conv1        │ Conv2d               │  1.2 M │ train │
        │ 71 │ model.layer4.0.bn1          │ BatchNorm2d          │  1.0 K │ train │
        │ 72 │ model.layer4.0.drop_block   │ Identity             │      0 │ train │
        │ 73 │ model.layer4.0.act1         │ ReLU                 │      0 │ train │
        │ 74 │ model.layer4.0.aa           │ Identity             │      0 │ train │
        │ 75 │ model.layer4.0.conv2        │ Conv2d               │  2.4 M │ train │
        │ 76 │ model.layer4.0.bn2          │ BatchNorm2d          │  1.0 K │ train │
        │ 77 │ model.layer4.0.act2         │ ReLU                 │      0 │ train │
        │ 78 │ model.layer4.0.downsample   │ Sequential           │  132 K │ train │
        │ 79 │ model.layer4.0.downsample.0 │ Conv2d               │  131 K │ train │
        │ 80 │ model.layer4.0.downsample.1 │ BatchNorm2d          │  1.0 K │ train │
        │ 81 │ model.layer4.1              │ BasicBlock           │  4.7 M │ train │
        │ 82 │ model.layer4.1.conv1        │ Conv2d               │  2.4 M │ train │
        │ 83 │ model.layer4.1.bn1          │ BatchNorm2d          │  1.0 K │ train │
        │ 84 │ model.layer4.1.drop_block   │ Identity             │      0 │ train │
        │ 85 │ model.layer4.1.act1         │ ReLU                 │      0 │ train │
        │ 86 │ model.layer4.1.aa           │ Identity             │      0 │ train │
        │ 87 │ model.layer4.1.conv2        │ Conv2d               │  2.4 M │ train │
        │ 88 │ model.layer4.1.bn2          │ BatchNorm2d          │  1.0 K │ train │
        │ 89 │ model.layer4.1.act2         │ ReLU                 │      0 │ train │
        │ 90 │ model.global_pool           │ SelectAdaptivePool2d │      0 │ train │
        │ 91 │ model.global_pool.pool      │ AdaptiveAvgPool2d    │      0 │ train │
        │ 92 │ model.global_pool.flatten   │ Flatten              │      0 │ train │
        │ 93 │ model.fc                    │ Linear               │  5.1 K │ train │
        │ 94 │ train_acc                   │ MulticlassAccuracy   │      0 │ train │
        │ 95 │ val_acc                     │ MulticlassAccuracy   │      0 │ train │
        │ 96 │ test_acc                    │ MulticlassAccuracy   │      0 │ train │
        └────┴─────────────────────────────┴──────────────────────┴────────┴───────┘

How to Train, Evaluate, and Infer Using Docker

 1. Build the Docker image:

          docker build -t dogbreed-classification .
 
 2. To run training:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/train.py

          Above docker script will generate the model artifact in below directory .
    
 <img width="470" alt="image" src="https://github.com/user-attachments/assets/6f518288-0ecd-4995-8eb9-4a73319d6f0f">
 
 3. To run evaluation:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/eval.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 
    
 4. To run inference:

          docker run -v $(pwd)/model_artifacts:/app/logs dogbreed-classification src/infer.py ckpt_path=./logs/train/runs/2024-10-08_10-50-43/checkpoints/epoch_001.ckpt

          please update the best model check point path generated from training script . 


 5 By default it performs inference on the images present in the dataset folder.

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

ouput from :  src/train.py
<img width="1075" alt="image" src="https://github.com/user-attachments/assets/6a844b1a-9553-4c19-8a00-9d6241ace6ec">
ouput from :  src/eval.py
<img width="1262" alt="image" src="https://github.com/user-attachments/assets/b56d2f54-71e9-4ce5-9a76-60aef51cfa10">
ouput from :  src/infer.py
<img width="1615" alt="image" src="https://github.com/user-attachments/assets/95dae635-e28f-4da6-afb5-6ceebc0e90bf">



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

  
    

