<h1>Dog Breed Classifier Deployment with EKS </h1>

This project demonstrates deploying a FastAPI-based CatDog Classifier application on a Kubernetes cluster using MiniKube. Follow these instructions to set up, deploy, and access the application.

<h2>Directory Structure:</h2>

        project/

        ├── app.py
        ├── fastapi_helm
        ├── model server
        ├── web server
        ├── templates/
        │   ├── index.html
        │   └── result.html
        ├── requirements.txt

<h2>Features</h2>

FastAPI-based model server

Docker Compose for local testing

Kubernetes manifests for deployment

Redis caching for inference performance

HELM chart for easy configuration and deployment

Ingress for external access
        

<h2>Architecture Diagram</h2>

![img1 drawio](https://github.com/user-attachments/assets/ad7dcd25-9889-4d0d-989f-33af0ef232ec)


<h3>Architecture Components:</h3>

<h3>Node: Minikube node running the cluster

<h3>Pods:</h3>

Model Server Pod: Hosts the deep learning model (FastAPI)

Redis Pod: Caches inference results

Web Server Pod: Frontend for user interaction

<h3>ReplicaSets:</h3> Ensures desired number of pods for scalability

<h3>Deployments:</h3> For managing pod lifecycle

<h3>Services:</h3>

Model Server Service: Exposes model server

Redis Service: Internal service for Redis communication

Web Server Service: Exposes web interface

<h3>Ingress:</h3> Routes external traffic to the web server

<h3>Volumes:</h3> Persistent storage for models and inference data

<h2>Helm</h2>
Deploying the Cat/Dog Classifier using Helm simplifies Kubernetes resource management by packaging all Kubernetes manifests into a single, reusable chart. This makes deployments more scalable, configurable, and easy to manage across environments.

<h3>Why Use Helm?</h3>

Reusability: Package Kubernetes resources into a single chart that can be deployed repeatedly.
Configurability: Easily override default values (like replica count, CPU, and memory limits) at deployment time.
Simplified Management: Manage deployments, upgrades, and rollbacks with simple Helm commands.


 <h3>Helm Commad : </h3>

 helm create fastapi-helm : Creates a new Helm chart for FastAPI.
 
 helm install fastapi-release-dev fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-dev.yaml : Deploys the chart for development with custom values.
 
 helm install fastapi-release-prod fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-prod.yaml : Deploys the chart for production with specific overrides.
 
 helm upgrade fastapi-release-dev fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-dev.yaml : Updates the development deployment with new changes.
 
 helm upgrade fastapi-release-prod fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-prod.yaml : Updates the production deployment.
 
 helm list : Lists all active Helm releases

 helm delete fastapi-release-prod : to delete helm already install 


<h2>k8s (Kubernetes)</h2>

k8s is a container orchestration system. It is used for container deployment and management. Its design is greatly impacted by Google’s internal system Borg.


<img width="538" alt="image" src="https://github.com/user-attachments/assets/c8d5e689-5805-4ed3-b5df-162af7b2a98c" />

A k8s cluster consists of a set of worker machines, called nodes, that run containerized applications. Every cluster has at least one worker node.

The worker node(s) host the Pods that are the components of the application workload. The control plane manages the worker nodes and the Pods in the cluster. In production environments, the control plane usually runs across multiple computers and a cluster usually runs multiple nodes, providing fault tolerance and high availability. 

Control Plane Components

API ServerThe API server talks to all the components in the k8s cluster. All the operations on pods are executed by talking to the API server.

SchedulerThe scheduler watches the workloads on pods and assigns loads on newly created pods.

Controller ManagerThe controller manager runs the controllers, including Node Controller, Job Controller, EndpointSlice Controller, and ServiceAccount Controller.

etcd etcd is a key-value store used as Kubernetes' backing store for all cluster data.

Nodes

PodsA pod is a group of containers and is the smallest unit that k8s administers. Pods have a single IP address applied to every container within the pod.

KubeletAn agent that runs on each node in the cluster. It ensures containers are running in a Pod.

Kube Proxykube-proxy is a network proxy that runs on each node in your cluster. It routes traffic coming into a node from the service. It forwards requests for work to the correct containers.

<h2>Amazon Elastic Kubernetes Service (EKS) Overview</h2>

Amazon Elastic Kubernetes Service (EKS) is a managed Kubernetes service that simplifies the deployment, management, and scaling of containerized applications. EKS integrates seamlessly with AWS services, offering features like scalability, security, and high availability, while reducing the operational complexity of running Kubernetes clusters.

Key Highlights:

Cluster Management:

Create, update, and delete Kubernetes clusters effortlessly using tools like eksctl and AWS Management Console.
Supports managed and self-managed node groups for a flexible compute setup.

Node Group Management:

Spot and on-demand instance support for cost efficiency and workload flexibility.
GPU instance support for compute-intensive applications like machine learning and graphics processing.

Add-Ons and Integrations:

Easily associate IAM roles and policies for enhanced security and permissions management.
Supports integrations with AWS services like Elastic Load Balancer (ELB), CloudWatch, and IAM.

Application Deployment:

Simplify application scaling with Kubernetes Horizontal Pod Autoscalers (HPA) and Cluster Autoscaler.
Utilize Helm charts for efficient application management and deployment.

Monitoring and Scaling:

Deploy tools like the Kubernetes metrics server for resource monitoring.
Enable cluster auto-scaling for optimized resource usage based on workload demands.
This README includes commands to manage EKS clusters, node groups, IAM service accounts, load balancers, and autoscaling configurations to help set up and maintain Kubernetes applications efficiently.


 <h3>Create Cluster:</h3>
   
    Creates a new EKS cluster based on a configuration file.
   
    eksctl create cluster -f eks-cluster.yaml
    
  <h3>Delete Cluster:</h3>
   
    Deletes an existing EKS cluster.
   
    eksctl delete cluster --name basic-cluster --region ap-south-1

  <h3>Delete Nodegroup:</h3>
   
    Deletes a specific nodegroup in the cluster.
   
    eksctl delete nodegroup --name ng-spot-4 --cluster basic-cluster 

   <h3>Create Nodegroup:</h3>
   
    Creates a new nodegroup using a configuration file.
   
    eksctl create nodegroup --config-file=eks-cluster.yaml   

   <h3>Delete Cluster (from config):</h3>
   
    Deletes the cluster specified in the configuration file.
   
    eksctl delete cluster -f eks-cluster.yaml 

   <h3>Associate IAM OIDC Provider:</h3>
   
    Associates an OIDC provider with the EKS cluster.
   
    eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster --approve
       
   <h3>Create IAM Policy:</h3>
   
    Creates an IAM policy for the LoadBalancer Controller.
   
    aws iam create-policy --policy-name AWSLoadBalancerControllerIAMPolicy --policy-document file://iam-policy.json 

  <h3>Create IAM Service Account for LoadBalancer Controller:</h3>
   
    Creates an IAM service account and associates the IAM policy.
   
    eksctl create iamserviceaccount \
            --cluster=basic-cluster \
            --namespace=kube-system \
            --name=aws-load-balancer-controller \
            --attach-policy-arn=arn:aws:iam::<accountid>:policy/AWSLoadBalancerControllerIAMPolicy \
            --override-existing-serviceaccounts \
            --region ap-south-1 \
            --approve


  <h3>Add Helm Repo:</h3>
   
    Adds the AWS EKS Helm charts repository.
   
    helm repo add eks https://aws.github.io/eks-charts
    helm repo update


  <h3>Install AWS Load Balancer Controller:</h3>
   
    Installs the AWS Load Balancer Controller using Helm.
   
    helm install aws-load-balancer-controller eks/aws-load-balancer-controller -n kube-system --set clusterName=basic-cluster --set serviceAccount.create=false --set 
    serviceAccount.name=aws-load-balancer-controller


  <h3>Create IAM Service Account for Cluster Autoscaler:</h3>
   
    Creates a service account for the Cluster Autoscaler.
   
    eksctl create iamserviceaccount \
    --cluster=basic-cluster \
    --namespace=kube-system \
    --name=cluster-autoscaler \
    --attach-policy-arn=arn:aws:iam::688567263021:policy/AWSClusterAutoScalerIAMPolicy \
    --override-existing-serviceaccounts \
    --region ap-south-1 \
    --approve


  <h3>Deploy Cluster Autoscaler:</h3>
   
    Applies the Cluster Autoscaler manifest for automatic scaling.
   
    wget https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
    kubectl apply -f cluster-autoscaler-autodiscover.yaml



  <h3>Monitor HPA:</h3>
   
    Watches the HPA status for web-server and model-server in the dev namespace.
   
    kubectl get hpa web-server-hpa --watch -n dev
    kubectl get hpa model-server-hpa --watch -n dev



  <h3>Describe HPA:</h3>
   
    Describes the HPA configuration for web-server and model-server
   
    kubectl describe hpa model-server-hpa
    kubectl describe hpa web-server-hpa

  <h3>Deploy FastAPI Application:</h3>
   
    Installs the FastAPI application using Helm with specific configurations.

   
    helm install fastapi-release-dev fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-dev.yaml



  <h3>Upgrade FastAPI Application:</h3>
   
    Upgrades the FastAPI Helm release with updated configurations.
   
    helm upgrade fastapi-release-dev fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-dev.yaml

  <h3>Delete FastAPI Application:</h3>
   
    Deletes the FastAPI Helm release.
   
    helm delete fastapi-release-dev fastapi-helm --values fastapi-helm/values.yaml -f fastapi-helm/values-dev.yaml

  <h3>Run Load Test:</h3>
   
    Executes a load test using the Python script test_requests.py. (need to update ing endpoint ) 
   
    python test_requests.py --requests 50000 --workers 50




<h3>Output of the following command present in logs folder </h3>
 
 kubectl describe <your_deployment>
 
 kubectl describe <your_pod>
 
 kubectl describe <your_ingress>
 
 kubectl top pod
 
 kubectl top node
 
 kubectl get all -o yaml


 <img width="1002" alt="image" src="https://github.com/user-attachments/assets/1898e11d-343c-4ef0-8f5d-899f31758ece" />
 
 <img width="1379" alt="image" src="https://github.com/user-attachments/assets/5d352639-61b1-41ac-8663-a666d985b5f1" />

 <img width="1437" alt="image" src="https://github.com/user-attachments/assets/7febe686-1f0d-4377-a559-7a8c88d14bd3" />

  <img width="1317" alt="image" src="https://github.com/user-attachments/assets/8e851423-7b1c-41d7-a6ab-50c4ee68801b" />

 <img width="970" alt="image" src="https://github.com/user-attachments/assets/4add3b07-3170-42c8-902a-01e47b6e00f7" />

 <img width="1014" alt="image" src="https://github.com/user-attachments/assets/d458ee1c-b181-4deb-bb70-c095247cb5c5" />

 <img width="1692" alt="image" src="https://github.com/user-attachments/assets/4b624e1d-c256-46d1-9d6d-06f65eec991c" />


 <img width="1432" alt="image" src="https://github.com/user-attachments/assets/3275a911-e4ca-4e34-82f5-b83f9c77e2c1" />

 <img width="1470" alt="image" src="https://github.com/user-attachments/assets/d6e70bcd-317f-402f-9fe9-23677140b3a1" />




 









   
