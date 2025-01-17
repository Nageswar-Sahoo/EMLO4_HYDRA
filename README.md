<h1>Dog Breed Classifier Deployment with EKS </h1>

This project demonstrates deploying a FastAPI-based CatDog Classifier application on a Kubernetes cluster using MiniKube. Follow these instructions to set up, deploy, and access the application.



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

<h2>EKS Cluster Operations</h2>

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





<h2>Kubernetes Commands</h2>
<h3>General Commands</h3></h4>

<h4>Get All Resources in a Namespace:</h4>

  kubectl get all -n <namespace>
<h4>Get Resource Details in YAML Format:</h4>

  kubectl get <resource-type> <resource-name> -o yaml
<h4>Apply a Configuration File:</h4>

  kubectl apply -f <file-name.yaml>
<h4>Delete a Resource:</h4>

  kubectl delete -f <file-name.yaml>
<h4>View Cluster Nodes:</h4>

  kubectl get nodes
<h3>Commands for Deployments</h3>
<h4>List Deployments:</h4>

kubectl get deployments
<h4>Describe a Deployment:</h4>

kubectl describe deployment <deployment-name>
<h4>Update a Deployment (Rolling Update):</h4>

kubectl set image deployment/<deployment-name> <container-name>=<new-image>
<h4>Scale a Deployment:</h4>

kubectl scale deployment/<deployment-name> --replicas=<number>
<h4>Restart a Deployment:</h4>

kubectl rollout restart deployment/<deployment-name>
<h4>Check Rollout Status:</h4>

kubectl rollout status deployment/<deployment-name>
<h4>Rollback a Deployment:</h4>

kubectl rollout undo deployment/<deployment-name>
<h3>Commands for Services</h3>
<h4>List Services:</h4>

kubectl get services
<h4>Describe a Service:</h4>

kubectl describe service <service-name>
<h4>Expose a Deployment as a Service:</h4>

kubectl expose deployment <deployment-name> --type=<type> --port=<port>
Example:

kubectl expose deployment catdog-classifier --type=NodePort --port=80
<h4>Access NodePort Service:</h4>

<h3>Commands for Ingress</h3>
<h4>List Ingress Rules:</h4>

kubectl get ingress
<h4>Describe an Ingress:</h4>

kubectl describe ingress <ingress-name>
<h4>Access Ingress: After applying the Ingress, check the external IP or host:</h4>

kubectl get ingress
Access it using the hostname or external IP in your browser.
<h4>Delete an Ingress:</h4>

kubectl delete ingress <ingress-name>
<h3>Commands for Pods</h3>
<h4>List Pods:</h4>

kubectl get pods
<h4>List Pods with Labels:</h4>

kubectl get pods -l <label-key>=<label-value>
<h4>Describe a Pod:</h4>

kubectl describe pod <pod-name>
<h4>Get Pod Logs:</h4>

kubectl logs <pod-name>
<h4></h4>Stream Pod Logs:</pod-name>

kubectl logs -f <pod-name>
<h4>Execute a Command Inside a Pod:</h4>

kubectl exec -it <pod-name> -- <command>
<h4>Delete a Pod:</h4>

kubectl delete pod <pod-name>
<h3>Namespace Management</h3>
<h4>List All Namespaces:</h4>

kubectl get namespaces
<h4>Create a New Namespace:</h4>

kubectl create namespace <namespace-name>
<h4>Set a Default Namespace:</h4>

kubectl config set-context --current --namespace=<namespace-name>
<h4>Delete a Namespace:</h4>

kubectl delete namespace <namespace-name>
<h3>Resource Debugging</h3>
<h4>Check Events in a Namespace:</h4>

kubectl get events -n <namespace>
<h4>Debug a Pod:</h4>

kubectl debug pod/<pod-name> -it --image=busybox
<h4>View Resource Usage:</h4>

  kubectl top pods
  kubectl top nodes

<h2>Project Setup</h2>

<h3>FastAPI Application Code</h3>
This is the core application handling image upload and classification. It uses FastAPI for API handling and Jinja2 for rendering HTML templates.

<h3>HTML Templates</h3>
These provide a user-friendly interface for uploading images and viewing results.


<h3>Dockerfile</h3>
Defines the containerization of the FastAPI application.

<h3>Kubernetes YAML Files</h3>

These configure the deployment, service, and ingress for your FastAPI application.


<h3>Building and Pushing the image to Minikube</h3>

Rebuild the Docker image to include the updated requirements:

eval $(minikube docker-env)

docker build -t fastapi-catdog-classifier .


<h3>Kubernetes Deployment</h3>

Start Minikube with the command: minikube start.

Navigate to the Kubernetes YAML files located in the k8s directory.

Use kubectl apply -f . to deploy the Kubernetes resources.

To remove the resources, run kubectl delete -f ..

<img width="1061" alt="image" src="https://github.com/user-attachments/assets/a07053e7-5d1e-4e01-9125-dbdc0328ff4c" />


<h3>How to access FAST API : </h3>

<img width="858" alt="image" src="https://github.com/user-attachments/assets/f7101312-288e-4b2d-9d1d-fc106c525e4b" />

<img width="1434" alt="image" src="https://github.com/user-attachments/assets/40d637b7-a199-4bb3-8ed3-d196043429dd" />

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

 <img width="970" alt="image" src="https://github.com/user-attachments/assets/4add3b07-3170-42c8-902a-01e47b6e00f7" />

 <img width="1014" alt="image" src="https://github.com/user-attachments/assets/d458ee1c-b181-4deb-bb70-c095247cb5c5" />

 <img width="1692" alt="image" src="https://github.com/user-attachments/assets/4b624e1d-c256-46d1-9d6d-06f65eec991c" />

 <img width="1432" alt="image" src="https://github.com/user-attachments/assets/3275a911-e4ca-4e34-82f5-b83f9c77e2c1" />

 <img width="1470" alt="image" src="https://github.com/user-attachments/assets/d6e70bcd-317f-402f-9fe9-23677140b3a1" />

 <img width="1317" alt="image" src="https://github.com/user-attachments/assets/8e851423-7b1c-41d7-a6ab-50c4ee68801b" />



 









   
