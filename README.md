<h1>Inference of a Dog Breed Image Dataset using Docker, AWS Lambda, Gradio with GitHub Actions. </h1>

This guide explains how to deploy a serverless AWS Lambda function using the AWS Cloud Development Kit (CDK). It includes setting up the environment, defining the Lambda function, and deploying the stack.


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


<h2>Project Setup</h2>
  <h3>1. Initialize the CDK Project </h3>
    
    Run the following command in your project directory to initialize a  CDK app:
    
    pip install aws-cdk-lib==2.168.0
    
    npm install -g aws-cdk
    
    Configure the AWS SDK with credentials for the IAM user created above
    
    Permissions needed by CDK
    
     {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "cloudformation:*",
                "ecr:*",
                "ssm:*",
                "s3:*",
                "iam:*"
            ],
            "Resource": "*"
        }
    ]
    }

 <h3>2. Define the Lambda Function and Resources </h3>

  This Python class, GradioLambda, defines an AWS CDK stack that deploys an AWS Lambda function built from a Docker image. The Lambda function is configured with specific memory, architecture, and timeout settings and is exposed via 
   an HTTPS Function URL with no authentication. Finally, the URL is outputted for easy access.

    class GradioLambda(Stack):
      def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create Lambda function
        lambda_fn = DockerImageFunction(
            self,
            "CatDogClassifier",
            code=DockerImageCode.from_image_asset(str(Path.cwd()), file="Dockerfile"),
            architecture=Architecture.X86_64,
            memory_size=3000,  # 8GB memory
            timeout=Duration.minutes(5),
        )

        # Add HTTPS URL
        fn_url = lambda_fn.add_function_url(auth_type=FunctionUrlAuthType.NONE)

        CfnOutput(self, "functionUrl", value=fn_url.url)

    app = App()
    gradio_lambda = GradioLambda(app, "GradioLambda", env=my_environment)
    app.synth()

  <h3>3. Bootstrap the CDK Environment </h3>
  
       Before deploying, bootstrap your AWS environment:  cdk bootstrap
    
  <h3>4. Synthesize the Stack </h3>
  
       Before deploying, synthesize your AWS environment:  cdk synthesize

  <h3>5. Deploy the Stack </h3>
  
        Deploy your resources using the following command: cdk deploy


<h2>Create a CI/CD Pipeline to deploy/update the model to AWS Lambda</h2>

  This GitHub Actions workflow automates the deployment of an AWS Lambda function using AWS CDK. It triggers on pushes or pull requests to the `feature/aws_lambda` branch, installs dependencies, configures AWS credentials, and runs 
   CDK commands (`bootstrap`, `synthesize`, and `deploy`) to deploy the Lambda function to AWS.

      name: AWS_LAMBDA_DEPLOYMENT
      steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Install Git LFS
        run: git lfs install

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'

      - name: Install Dependencies
        run: |
          pip install boto3 aws-cdk-lib==2.168.0
          npm install -g aws-cdk

      - name: Configure AWS Credentials
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
        run: |
          aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
          aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
          aws configure set region $AWS_REGION

      - name: CDK Deploy
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
        run: |
          cdk bootstrap
          cdk synthesize
          cdk deploy --require-approval never

<img width="1447" alt="image" src="https://github.com/user-attachments/assets/f0e285e3-e0ce-416c-86c6-f60f06e7f4a6" />

<h3> Inference Link :  </h3>   https://mq6lgqox7zlntc67xaeqlmd53e0itkgy.lambda-url.ap-south-1.on.aws/
<h3>Prediction Results In Gradio </h3>

<img width="1585" alt="image" src="https://github.com/user-attachments/assets/91e90cc3-d4c4-41d9-8085-607e4827046c">
<img width="1616" alt="image" src="https://github.com/user-attachments/assets/c607d9ab-d256-4ddd-ae35-9216ad3b3c0b">



<h3>Requirements</h3>

         Docker
         Kaggle API (for downloading the dataset)
         GitHub Codespaces or Visual Studio Code with the Remote Containers extension (for DevContainer setup)

  
    

