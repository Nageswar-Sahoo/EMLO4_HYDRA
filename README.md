<h1>Dog Breed Image Dataset Training, Evaluation, and Inference with Docker And AWS with Github Action </h1>

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


<h2>Setting Up S3 as a DVC Remote for Data Storage</h2>

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
     
<img width="1788" alt="image" src="https://github.com/user-attachments/assets/5488b75d-7e02-48d4-b157-d6c381510751">

<h2>GitHub Actions Workflow for Building and Pushing Docker Image to Amazon ECR</h2>


This guide provides instructions to set up a GitHub Actions workflow for building a Docker image and pushing it to Amazon ECR (Elastic Container Registry). This is useful for automating the deployment of Docker images in AWS.

<h4>Prerequisites</h4>

    1. AWS Account: Ensure you have an AWS account and necessary permissions for ECR.
    
    2. ECR Repository: Set up an Amazon ECR repository in your AWS console to store your Docker images.
    
    3. GitHub Repository: A repository where you can configure GitHub Actions.
    
<h3>Steps to Set Up GitHub Actions</h3>

<h4>Add Workflow Configuration: Copy the following code to your workflow file. This configuration will:</h4>

   1.Build a Docker image from your repository

   2.Tag the image

   3.Log in to Amazon ECR

   4.Push the Docker image to your specified ECR repository



        build-and-push-image:
        needs: dvc
        runs-on: ubuntu-latest
        permissions:
          contents: read
          packages: write
    
        steps:
        - name: Checkout repository
          uses: actions/checkout@v4
    
        - name: Configure AWS Credentials
          run: |
            aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
            aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
            aws configure set region ${{ secrets.AWS_REGION }}

        - name: Login to Amazon ECR
          id: login-ecr
          uses: aws-actions/amazon-ecr-login@v1
                      
        - name: Get commit hash
          id: get-commit-hash
          run: echo "::set-output name=commit-hash::$(git rev-parse --short HEAD)"
        - name: Get timestamp
          id: get-timestamp
          run: echo "::set-output name=timestamp::$(date +'%Y-%m-%d-%H-%M')"
  
        - name: Build, tag, and push the image to Amazon ECR
          id: build-image
          env:
            ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
            ECR_REPOSITORY: ${{ secrets.REPO_NAME }}
            IMAGE_TAG: ${{ steps.get-commit-hash.outputs.commit-hash }}-${{ steps.get-timestamp.outputs.timestamp }}
          run: |
            docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
            docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG


<h3>Explanation of Workflow Steps</h3>
<h4>1.Checkout Repository:</h4>

    Uses the actions/checkout@v4 action to pull the repository code to the GitHub Actions runner.

<h4>2.Configure AWS Credentials:</h4>

    Configures AWS CLI with the necessary credentials to access AWS services.
    The AWS access key, secret access key, and region are stored as GitHub secrets (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION).

<h4>3.Login to Amazon ECR:</h4>

     Uses the aws-actions/amazon-ecr-login@v1 action to authenticate the GitHub Actions runner with Amazon ECR.
<h4>4.Get Commit Hash and Timestamp:</h4>

            Retrieves the latest commit hash and timestamp to create a unique tag for the Docker image. 
	    This ensures each image is uniquely identifiable.

<h4>5.Build, Tag, and Push Docker Image:</h4>

             Builds the Docker image using the Dockerfile in the repository.
             Tags the image with the ECR registry URL, ECR repository name, commit hash, and timestamp.
             Pushes the image to the specified Amazon ECR repository.      


<h4>Setting Up GitHub Secrets</h4>
To ensure security and proper access, add the following secrets to your GitHub repository under 

Settings > Secrets and variables > Actions:

AWS_ACCESS_KEY_ID: Your AWS access key.

AWS_SECRET_ACCESS_KEY: Your AWS secret key.

AWS_REGION: The AWS region of your ECR repository (e.g., us-east-1).

REPO_NAME: The name of your Amazon ECR repository.

<h4>Triggering the Workflow</h4>
This workflow will trigger on every push to the main branch. You can adjust the triggering branch by modifying the branches section in the on: push configuration.

<h4>Viewing the Image in Amazon ECR</h4>
Once the workflow completes, navigate to your Amazon ECR repository in the AWS Console. You should see a new image tagged with the commit hash and timestamp.
<img width="1470" alt="image" src="https://github.com/user-attachments/assets/47d42e23-90ef-4dc5-8c0b-9a2519db0421">

<h2>GitHub Self-Hosted Runner on EC2</h2>

Continuous Integration (CI) and Continuous Deployment (CD) have become essential practices in modern software development, streamlining development processes and accelerating delivery. GitHub Actions offers a powerful CI/CD platform, and one of its standout features is the use of self-hosted runners. These runners allow you to run workflows on infrastructure you control, such as EC2 instances, providing greater flexibility, cost savings, and control over your environment. In this blog post, we will explore what self-hosted runners are, their advantages, and provide a step-by-step guide to setting up an EC2 Ubuntu instance as a self-hosted runner for your GitHub workflows.


<h3>Adding an EC2 Ubuntu Instance as a Self-Hosted Runner:</h3>
Now, let’s walk through the steps to add an EC2 Ubuntu instance as a self-hosted runner on GitHub:

<h4>Step 1: Set Up an EC2 Instance on AWS</h4>

Log in to the AWS Management Console.
Navigate to the EC2 dashboard and launch a new Ubuntu instance.
Ensure that the instance has the necessary network configurations, security groups, and key pairs.

<img width="1533" alt="image" src="https://github.com/user-attachments/assets/1807a49d-eb62-49cb-b998-5f079ddf5aa4">



<h4>Step 2: Install GitHub Runner on the EC2 Instance</h4>
  
  Connect to your EC2 instance using SSH.
  Download the GitHub Actions self-hosted runner package from the GitHub repository.
  Extract the downloaded package.
  Run the configuration script and follow the prompts.
  Replace your-username and your-repository with your GitHub username and repository name.

  
  Create a folder
  mkdir actions-runner && cd actions-runner
 
  Download the latest runner package
  curl -o actions-runner-linux-x64-2.320.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.320.0/actions-runner-linux-x64-2.320.0.tar.gz
 
  Extract the installer
  tar xzf ./actions-runner-linux-x64-2.320.0.tar.gz
 
  Create the runner and start the configuration experience
  ./config.sh --url https://github.com/Nageswar-Sahoo/EMLO4_HYDRA --token <token>
 


<h4>Step 3: Start the Self-Hosted Runner Service </h4>

  Copied!# Last step, run it!
  ./run.sh

<h4>Step 4: Verify and Monitor the Runner</h4>

Go to your GitHub repository, navigate to “Settings,” and click on “Actions.”
You should see your self-hosted runner listed under the “Runners” section.

<img width="1488" alt="image" src="https://github.com/user-attachments/assets/627b87ed-11fb-48c2-b48a-b890162dd07f">


<h4>Step 5: Use Self-Hosted Runners in Your Workflow:</h4>

Once your self-hosted runner is set up, integrating it into your GitHub Actions workflow is a breeze. Simply add the following line to your workflow YAML file:

        jobs:
          pull-and-run-image:
            runs-on: self-hosted


 <img width="1341" alt="image" src="https://github.com/user-attachments/assets/a880bb33-c4f3-4f4b-be77-062f1f340706">
 

<img width="1739" alt="image" src="https://github.com/user-attachments/assets/ca1ca4bb-b8c9-41c5-9001-4947d302d163">




<h2>Run Docker Image from ECR with GitHub Actions</h2>

This GitHub Actions workflow pulls a specified Docker image from Amazon ECR and runs it with a mounted volume for logging. The workflow is triggered manually (workflow_dispatch) and accepts the ecr_image_name as input. It authenticates to AWS using configured secrets, pulls the image, and runs it with the local model_artifacts directory mounted to /app/logs inside the container. Logs are optionally displayed after execution.


           pull-and-run-image:
              runs-on: self-hosted
              permissions:
                contents: read
                packages: write

              steps:
                - name: Grant Execute Permissions to Script
                  run: chmod +x src/train.py
                # Step 1: Configure AWS Credentials
                - name: Configure AWS Credentials
                  run: |
                    aws configure set aws_access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
                    aws configure set aws_secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
                    aws configure set region ${{ secrets.AWS_REGION }}

                # Step 2: Login to Amazon ECR
               - name: Login to Amazon ECR
                  id: login-ecr
                  uses: aws-actions/amazon-ecr-login@v1

                # Step 3: Pull the Docker Image
                - name: Pull Docker Image
                  env:
                    ECR_IMAGE_NAME: ${{ github.event.inputs.ecr_image_name }}
                    ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
                  run: |
                    FULL_IMAGE_URI="${ECR_REGISTRY}/${ECR_IMAGE_NAME}"
                    echo "Pulling Docker image: ${FULL_IMAGE_URI}"
                    docker pull "${FULL_IMAGE_URI}"

                # Step 4: Run the Docker Image with Volume and Script
                - name: Run Docker Image
                  env:
                    ECR_IMAGE_NAME: ${{ github.event.inputs.ecr_image_name }}
                    ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
                  run: |
                    FULL_IMAGE_URI="${ECR_REGISTRY}/${ECR_IMAGE_NAME}"
                    WORKSPACE="$(pwd)"
                    echo "Running Docker image: ${FULL_IMAGE_URI} with script ${SCRIPT_PATH}"
                    docker run \
                      -v "${WORKSPACE}/model_artifacts:/app/logs" \
                      "${FULL_IMAGE_URI}"
                # Step 5: Display Logs (Optional - If logs are stored in a volume)

<img width="1436" alt="image" src="https://github.com/user-attachments/assets/24230e8a-b0bf-4872-9156-5eda9568f6cc">

<h2>Using Amazon S3 for Model Checkpoints</h2>

Amazon S3 (Simple Storage Service) is a highly scalable and durable object storage service provided by AWS. It is commonly used for storing and managing large datasets, backups, and model checkpoints in machine learning workflows. By storing model checkpoints on S3, you can ensure that your models are securely backed up and easily accessible for future inference or further training.

In machine learning projects, model checkpoints are saved at various stages of training to allow recovery from interruptions, fine-tuning, or evaluation at later stages. Using S3 to store these checkpoints enables seamless integration with cloud-based infrastructure, making it easy to manage and retrieve checkpoints from different environments.

By integrating AWS CLI or SDKs into your training pipeline, you can automatically upload checkpoints to an S3 bucket during the training process. This approach improves data accessibility, security, and scalability.

<img width="1759" alt="image" src="https://github.com/user-attachments/assets/98c52f96-1296-451f-bd90-107f5628c09c">


<h2>How to Train, Evaluate, and Infer Using Docker</h2>

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

  
    

