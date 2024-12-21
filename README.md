<h1>Dog Breed Classifier Deployment with Kubernetes and MiniKube </h1>



This guide explains how to deploy a serverless AWS Lambda function using the AWS Cloud Development Kit (CDK). It includes setting up the environment, defining the Lambda function, and deploying the stack.

<h2>Directory Structure:</h2>

        project/

        ├── app.py
        ├── templates/
        │   ├── index.html
        │   └── result.html
        ├── requirements.txt
        └── Dockerfile

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

minikube service <service-name>
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


<h2>MiniKube</h2>
<h3>Start MiniKube</h3>

<h3>1. Start MiniKube</h3>
MiniKube is a local Kubernetes environment. Use it to test Kubernetes deployments on your local machine.

Steps:
Ensure MiniKube is installed. If not, install it following the MiniKube installation guide.

Start MiniKube:
    minikube start --cpus=4 --memory=8192
    This starts a MiniKube cluster with 4 CPUs and 8 GB of memory.

Verify that MiniKube is running:
   minikube status

<h2>MiniKube Commands</h2>
<h4>Start MiniKube:</h4>
  minikube start
<h4>Stop MiniKube:</h4>
   minikube stop
<h4>Delete MiniKube Cluster:</h4>
  minikube delete
<h4>Access Kubernetes Dashboard:</h4>
  minikube dashboard
<h4>Enable Add-ons (e.g., ingress):</h4>
   minikube addons enable ingress
   
<h4>Tunnel to the Ingress</h4>

MiniKube does not expose Ingress directly on your host machine. Use the MiniKube tunnel to expose the Ingress.

Start a MiniKube tunnel in a separate terminal:

   minikube tunnel

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



   
