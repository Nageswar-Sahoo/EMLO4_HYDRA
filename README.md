<h1>Training, Evaluation, and Inference of a Dog Breed Image Dataset using Docker, AWS, Gradio, and HuggingFace, Tracing integrated with GitHub Actions. </h1>

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

<h2>Training The Model : Run Docker Image from ECR with GitHub Actions</h2>

This GitHub Actions workflow pulls a specified Docker image from Amazon ECR and runs it with a mounted volume for logging. The workflow is triggered manually (workflow_dispatch) and accepts the ecr_image_name as input. It authenticates to AWS using configured secrets, pulls the image, and runs it with the local model_artifacts directory mounted to /app/logs inside the container. Logs are optionally displayed after execution.

      - name: Run the Docker container
        env:
         ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
         ECR_REPOSITORY: ${{ secrets.REPO_NAME }}
         IMAGE_TAG: ${{ steps.get-commit-hash.outputs.commit-hash }}-${{ steps.get-timestamp.outputs.timestamp }}
         AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
         AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
         AWS_REGION: ${{ secrets.AWS_REGION }}
        run: |
          FULL_IMAGE_URI="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"
          WORKSPACE="$(pwd)"
        
          echo "Pulling Docker image: ${FULL_IMAGE_URI}"
          docker pull "${FULL_IMAGE_URI}"
        
          echo "Running Docker image: ${FULL_IMAGE_URI}"
          docker run \
          --rm \
          -v "${WORKSPACE}/model_artifacts:/app/logs" \
          -e AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID}" \
          -e AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY}" \
          -e AWS_REGION="${AWS_REGION}" \
          "${FULL_IMAGE_URI}"

<img width="1376" alt="image" src="https://github.com/user-attachments/assets/a7da7046-017a-4ad1-b086-284fcf3a12b6">

<h2>Using Amazon S3 for Model Checkpoints</h2>

Amazon S3 (Simple Storage Service) is a highly scalable and durable object storage service provided by AWS. It is commonly used for storing and managing large datasets, backups, and model checkpoints in machine learning workflows. By storing model checkpoints on S3, you can ensure that your models are securely backed up and easily accessible for future inference or further training.

In machine learning projects, model checkpoints are saved at various stages of training to allow recovery from interruptions, fine-tuning, or evaluation at later stages. Using S3 to store these checkpoints enables seamless integration with cloud-based infrastructure, making it easy to manage and retrieve checkpoints from different environments.

By integrating AWS CLI or SDKs into your training pipeline, you can automatically upload checkpoints to an S3 bucket during the training process. This approach improves data accessibility, security, and scalability.

<img width="1788" alt="image" src="https://github.com/user-attachments/assets/cf0feb1e-fc9c-410c-8886-3fe0de56289a">
<img width="1425" alt="image" src="https://github.com/user-attachments/assets/c3544264-c85e-467d-9dcc-c76d951fa26d">

<h1>Deploying The Model</h1>

<h2>TorchScript</h2>

TorchScript is an intermediate representation of PyTorch models that allows you to save, serialize, and run models independently of Python. It bridges the gap between research and production by enabling efficient deployment of PyTorch models in environments where Python runtime may not be available.

Key features include:

Serialization: Save models as .pt files for portability.
Execution: Run models with optimized performance in C++ or other runtimes.
Flexibility: Convert models written in standard PyTorch seamlessly using tracing or scripting.
TorchScript makes it easier to transition from experimentation to deployment while preserving PyTorch's dynamic and flexible nature.

TorchScript provides two ways to convert a PyTorch model into its intermediate representation: Scripting and Tracing. Here's a comparison to help understand the differences and when to use each:

<h3>1. Scripting</h3>
What It Is:
Scripting involves directly converting a PyTorch nn.Module or function into TorchScript by analyzing its Python code.
It captures the entire logic, including control flow (if-else, loops).

<h4>Advantages:</h4>

Handles dynamic control flows (e.g., loops, conditionals).
Fully preserves the logic of the original Python code.
Suitable for models with complex computations.

<h4>Disadvantages:</h4>

Requires code to follow certain TorchScript compatibility rules.
Slightly more effort to debug due to strict type checking.

<h3>2. Tracing</h3>

What It Is:
Tracing records the operations executed during a single run of the model with example inputs.
It produces a static computational graph.

<h4>Advantages:</h4>

Quick and straightforward for static models without control flow.
Easier to apply when the model's structure does not change with inputs.

<h4>Disadvantages:</h4>

Ignores dynamic control flows (e.g., if-else, loops); these are "baked in" during tracing.
Requires careful testing to ensure the traced model behaves correctly for all inputs.

<h2>Why Use TorchScript?</h2>
TorchScript simplifies and accelerates the transition from research to production by:

Enabling optimized inference.

Supporting deployment on edge devices and servers.

Reducing Python-specific dependencies for secure and scalable deployment.

<h2>Getting Started</h2>

Convert your PyTorch model to TorchScript using tracing or scripting.

Save the TorchScript model using torch.jit.save().

Deploy it using the Python or C++ runtime for efficient production integration.

TorchScript is your solution for reliable, high-performance model deployment in the modern AI landscape.

<img width="1426" alt="image" src="https://github.com/user-attachments/assets/f1dcf5b0-3010-418b-9d99-a7508d1504ed">

<h2>Gradio</h2>

Gradio is an open-source Python library that simplifies the creation of interactive user interfaces (UIs) for machine learning models, APIs, and other Python-based applications. It enables you to build web-based UIs with minimal code, making it easy to showcase and test models in real-time.

<h3>Key Features</h3>

Ease of Use: Quickly create UIs with just a few lines of code.

Interactive Components: Provides pre-built inputs (e.g., text, image, audio) and outputs for seamless integration.

Web-Based: Automatically generates a local or shareable web interface.

Customizable: Easily modify components to suit your application's needs.

Integration: Works with popular frameworks like PyTorch, TensorFlow, Hugging Face, and more.

<h2>Hugging Face Spaces</h2>

Hugging Face Spaces is a free, collaborative platform for hosting and sharing machine learning demos and applications. It supports popular frameworks like Gradio and Streamlit, making it easy to build, deploy, and showcase interactive ML models and tools.

<h3>Key Features</h3>

User-Friendly Deployment: Host apps with minimal effort.

Framework Support: Compatible with Gradio, Streamlit, and static HTML/JS.

Community Sharing: Share your projects with the Hugging Face community.

Free Hosting: Public spaces are hosted for free with GPU options available.

<h2>Why Use Hugging Face Spaces?</h2>

Showcase your ML models to a global audience.

Collaborate and gather feedback easily.

Streamline the deployment of interactive demos.

<h2>Hugging Face Spaces and Gradio</h2>

Hugging Face Spaces and Gradio together form a powerful combination for building, hosting, and sharing interactive machine learning demos effortlessly.

<img width="1430" alt="image" src="https://github.com/user-attachments/assets/f530b621-b963-4be0-ab56-323fdb72fe2a">

<img width="1410" alt="image" src="https://github.com/user-attachments/assets/d6810603-942c-473c-9f30-e4834a3ec18e">

<h3>Hugging Face Spaces Deployment Link : </h3> https://huggingface.co/spaces/Nageswar-250/Dog_Breed_Classifier




<h3>Prediction Results In Gradio </h3>

<img width="1585" alt="image" src="https://github.com/user-attachments/assets/91e90cc3-d4c4-41d9-8085-607e4827046c">
<img width="1616" alt="image" src="https://github.com/user-attachments/assets/c607d9ab-d256-4ddd-ae35-9216ad3b3c0b">






<h3>Requirements</h3>

         Docker
         Kaggle API (for downloading the dataset)
         GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

  
    

