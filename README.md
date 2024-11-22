<h1>Assignment-1 : Dog Breed Image Dataset Inference with LitServe </h1>

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


<h2>LitServer: Fast, Simple, and Scalable Model Serving </h2>

LitServer is a lightweight, high-performance model-serving framework designed to make deploying machine learning models easy and efficient. Built on Lightning AI, LitServer provides a straightforward way to serve deep learning and machine learning models as RESTful APIs with minimal configuration. Whether you're deploying a computer vision model, natural language processing pipeline, or custom machine learning workflow, LitServer helps streamline model deployment with a focus on speed, scalability, and ease of use.

<h3>Key Features</h3>
<h4>Quick Deployment: </h4> Easily deploy machine learning models without complex setup. LitServer supports frameworks like PyTorch, TensorFlow, and Hugging Face Transformers out of the box.

<h4>High-Performance Inference: </h4> Built with optimized processing to handle high-throughput and low-latency requests, making it ideal for production environments.

<h4>Scalability:  </h4>Supports concurrent request handling with configurable worker threads and process scaling, allowing you to handle multiple requests efficiently.

<h4>FP16 Support: </h4> LitServer can serve models in half-precision (FP16) mode, which reduces memory usage and improves inference speed on compatible hardware.

<h4>Customizable API:  </h4> LitServer allows you to define custom routes and request handling, giving you control over how the model serves and responds to requests.

<h2>Assignment-1 </h2>


<h2>LitServe Request Handling </h2>

LitServe, part of the Lightning AI ecosystem, provides a framework for serving machine learning models and handling inference requests. To handle concurrent requests effectively in LitServe, you'll need to leverage scaling, multi-threading, and possibly load balancing to ensure the model server can manage multiple requests at once.

Here are some ways to handle concurrent requests in LitServe:

<h3>1. Default Request And Response Handling </h3>

LitServe does not handle multiple requests by default . LitServe requires additional configuration to efficiently handle multiple concurrent requests, especially if you are running inference jobs or serving models.

<img width="1413" alt="image" src="https://github.com/user-attachments/assets/883edc6d-223b-4d65-b98c-466c9368e039">

![image](https://github.com/user-attachments/assets/c732908c-5fc0-4f55-a95b-bd1919b23a22)

![image](https://github.com/user-attachments/assets/43d0a2ab-61e9-4551-9427-d14ddf12ecc0)


By default, LitServe may be single-threaded, meaning it processes requests one by one on a single worker process or thread this is one of the primary reasion for less Requests per Second and CPU Usage . 

Server Config : 

      server = ls.LitServer(
        api,
        accelerator="gpu",
       )
       server.run(port=8000)


<h3>2. Batching Requests for Efficiency</h3>
If your model supports batching (processing multiple inputs at once), you can configure LitServe to batch incoming requests. Batching allows you to process multiple inference requests simultaneously, which improves performance and reduces processing time for each individual request.

<img width="1417" alt="image" src="https://github.com/user-attachments/assets/791b032b-6752-455a-8d07-d7cc4a714008">

![image](https://github.com/user-attachments/assets/5e2778b1-9bbd-4ff2-9557-703c1ee04707)

![image](https://github.com/user-attachments/assets/77ebcd25-01bc-434a-8820-b3ca92e31e7f)

As LitServe queues requests for a specified duration and processes them collectively, we observe a slight improvement in performance in terms of Requests per Second and CPU usage.

Server Config : 

    server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  
        batch_timeout=0.01,  
    )
    server.run(port=8000)

<h3>3. Use Multiple Workers with LitServe</h3>

LitServe supports running the server with multiple workers, This allows the server to spin up separate processes, each capable of handling requests independently. 
. Each worker handles incoming requests concurrently, improving throughput and ensuring that the server can handle multiple requests without blocking.
You can configure multiple workers in the LitServe configuration to enable parallel processing.

<img width="1408" alt="image" src="https://github.com/user-attachments/assets/93058fca-1e3c-4036-8229-46c4ba854f6c">

![image](https://github.com/user-attachments/assets/ae359b3f-36c6-4799-9b45-718370a2af18)

![image](https://github.com/user-attachments/assets/f0e91db7-0114-4839-b4c8-e55ae6dc58f5)

With multiple workers running as separate server instances, the system can handle multiple requests per instance. As each instance processes requests, there is a noticeable improvement in performance, with Requests per Second increasing. However, CPU usage is also significantly higher, reaching levels of 80-90% or more.

Server Config : 

      server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  # Adjust based on your GPU memory and requirements
        batch_timeout=0.01,  # Timeout in seconds to wait for forming batches
        workers_per_device=4
       )
       server.run(port=8000)

<h3>4. Half Precision (FP16)</h3>

Half Precision (FP16) in LitServe refers to using 16-bit floating-point numbers (instead of the standard 32-bit floating-point numbers or FP32) for model inference. Using half-precision can significantly reduce memory usage and speed up inference, especially on GPUs that support FP16 operations, without sacrificing much accuracy for many types of models.

<img width="1425" alt="image" src="https://github.com/user-attachments/assets/cfb16eee-5d14-48b7-9f74-44e1772a7028">

![image](https://github.com/user-attachments/assets/88536a2d-b024-49dc-abde-569d5986e498)

![image](https://github.com/user-attachments/assets/f7b9d9dd-d90a-4e18-8b7d-15c83980ee2e)


Using 16-bit floating-point numbers (half-precision) can significantly reduce memory usage and accelerate inference, with minimal impact on accuracy for many model types. This improvement is evident from the benchmark results.

Server Config : 

        self.model = self.model.to(device).to(precision)
	torch.stack(batched_tensors).to(self.device).to(precision)

     server = ls.LitServer(
        api,
        accelerator="gpu",
        max_batch_size=64,  # Adjust based on your GPU memory and requirements
        batch_timeout=0.01,  # Timeout in seconds to wait for forming batches
        workers_per_device=4
       )
    server.run(port=8000)

<h3>5. Load Balancing Across Multiple LitServe Instances</h3>

To handle even higher levels of concurrency, you can deploy multiple LitServe instances (possibly in different regions or clusters) and configure a load balancer (like Nginx or HAProxy) or use a cloud-based load balancer (AWS ELB, Google Cloud Load Balancer, etc.) to distribute traffic across LitServe instances.

This helps in scaling horizontally and ensures that no single instance is overloaded.

<h2>Assignment-2 : Deploy any llama based llm (meta-llama/Llama-3.2-1B-Instruct) </h2>

The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. 

        checkpoint = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the desired LLaMA checkpoint
        self.device = device
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)  # LLaMA models often require `use_fast=False`
        if self.tokenizer.pad_token is None:
           self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token if available
        self.model = model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.model.eval()

<h3>1. Default Request And Response Handling </h3>

<img width="1399" alt="image" src="https://github.com/user-attachments/assets/8b732aa4-cd62-4ebf-8bf7-6e643e4f10dc">

![image](https://github.com/user-attachments/assets/d1b4e574-2c4b-4d59-9021-1f973c91331a)

<img width="1423" alt="image" src="https://github.com/user-attachments/assets/4206502b-3d8b-4c39-ab60-e6f8eac3cca1">

![image](https://github.com/user-attachments/assets/fd4996b8-618e-435c-be36-ba97e3434d7f)


<h3>3. Use Multiple Workers with LitServe</h3>

I have configure 2 workers in the LitServe configuration to enable parallel processing.

<img width="1398" alt="image" src="https://github.com/user-attachments/assets/68a6047e-7aa9-494a-ac60-39fdbd71830d">

![image](https://github.com/user-attachments/assets/ff8ff1d3-3d82-41c1-85f1-95415cdd8855)

<img width="1448" alt="image" src="https://github.com/user-attachments/assets/25d4881a-c301-434c-8631-e7f94f5b67af">


![image](https://github.com/user-attachments/assets/f48b934c-a774-4b01-8bbc-548d079c234a)




<h3>4. Half Precision (FP16)</h3>

<img width="1425" alt="image" src="https://github.com/user-attachments/assets/c63c7738-8e96-4e7a-a80d-9d65c6e7c74a">

![image](https://github.com/user-attachments/assets/15cc18a7-126b-4369-bd54-b4a027fbe913)


<img width="1415" alt="image" src="https://github.com/user-attachments/assets/03564ba2-1b8d-41f8-bf99-5dbee4298274">

![image](https://github.com/user-attachments/assets/058b24c1-0782-4082-b5f5-32b8997c335a)







