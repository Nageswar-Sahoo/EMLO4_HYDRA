# EKS Cluster and Kubernetes Setup Guide

This README provides a step-by-step guide to setting up an **Amazon EKS Cluster**, configuring IAM roles, installing **Istio**, **KServe**, **AWS Load Balancer Controller**, and **NVIDIA GPU Operator** using **Helm**. Each section contains detailed explanations, verification steps, and best practices.

---

## Introduction to Model Serving with SD3, Istio, and KServe

### **SD3 Model**
The **Stable Diffusion 3 (SD3) model** is an advanced deep learning model designed for high-quality image generation. It leverages diffusion-based techniques to generate detailed and coherent images from text prompts. When deploying SD3 in a cloud-native environment, efficient scaling, resource management, and low-latency inference are critical. This is where **Istio** and **KServe** play a crucial role in optimizing the model serving process.

### **Istio for Model Serving**
**Istio** is a powerful service mesh that provides traffic management, security, and observability for microservices, including ML model serving workloads. When used for model deployment:
- **Traffic Routing:** It ensures smooth request handling across multiple model versions.
- **Load Balancing:** Distributes inference requests efficiently across pods.
- **Security & Authentication:** Supports mutual TLS (mTLS) and role-based access control (RBAC) for secure API interactions.
- **Observability:** Enables detailed monitoring through tools like **Kiali** and **Prometheus**.

### **KServe for Model Inference**
**KServe** (formerly KFServing) is a Kubernetes-based serving tool tailored for ML models. It simplifies and automates model deployment while ensuring high availability and scalability. Key features include:
- **RawDeployment & InferenceService:** Supports deploying models as individual Kubernetes services or as Istio-managed inference services.
- **Multi-Framework Support:** Works with TensorFlow, PyTorch, ONNX, and custom ML models like SD3.
- **Autoscaling:** Uses **Knative** to scale model replicas based on request traffic.
- **Canary Deployment:** Enables A/B testing and gradual rollout of new model versions.

By integrating **Istio and KServe**, organizations can deploy **SD3 and other deep learning models** in a robust, scalable, and efficient manner on **Amazon EKS**.

---

## 1. Amazon EKS Cluster Setup

### **Creating an EKS Cluster**
```bash
eksctl create cluster -f eks-cluster.yaml
```
- This command creates an Amazon EKS cluster based on the configuration specified in `eks-cluster.yaml`.
- Ensure that the YAML file includes details like **region, VPC settings, node groups, and IAM roles**.
- Verify the cluster creation with:
  ```bash
  kubectl get nodes
  ```

### **Deleting an EKS Cluster**
```bash
eksctl delete cluster -f eks-cluster.yaml
```
- This will delete the entire EKS cluster along with associated resources.
- Before deletion, backup any important data such as persistent volumes and configurations.

---

## 2. Configuring IAM Roles and Policies

### **Associating IAM OIDC Provider with EKS**
```bash
eksctl utils associate-iam-oidc-provider --region ap-south-1 --cluster basic-cluster12 --approve
```
- This step is required to allow Kubernetes service accounts to use AWS IAM roles.
- Verify association with:
  ```bash
  aws eks describe-cluster --name basic-cluster12 --query "cluster.identity.oidc.issuer" --output text
  ```

### **Creating an IAM Policy for S3 Access**
```bash
aws iam create-policy \
    --policy-name S3ListTestEMLO \
    --policy-document file://iam-s3-test-policy.json
```
- This policy grants permissions to access specific S3 buckets.
- Verify policy creation with:
  ```bash
  aws iam list-policies --query "Policies[?PolicyName=='S3ListTestEMLO']"
  ```

### **Deleting an IAM Policy**
```bash
aws iam delete-policy --policy-arn arn:aws:iam::<account-id>:policy/S3ListTestEMLO
```
- Ensure that no roles or service accounts are attached before deleting.

---

## 3. Creating IAM Service Accounts for Kubernetes

### **Creating a Service Account for S3 Access**
```bash
eksctl create iamserviceaccount \
  --name s3-list-sa \
  --cluster basic-cluster12 \
  --attach-policy-arn arn:aws:iam::688567263021:policy/S3ListTestEMLO \
  --approve \
  --region ap-south-1
```
- Creates a **Kubernetes service account** and attaches an IAM policy to it.
- Verify service account creation with:
  ```bash
  kubectl get serviceaccount s3-list-sa -n default
  ```

---

## 4. Installing NVIDIA GPU Operator

### **Adding NVIDIA Helm Repository**
```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
```
- Adds the official NVIDIA Helm chart repository for GPU management.

### **Installing the GPU Operator**
```bash
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version=v24.9.1
```
- This deploys the **NVIDIA GPU Operator**, which enables GPU scheduling and monitoring.
- Verify GPU node detection with:
  ```bash
  kubectl get nodes -o json | jq '.items[].status.allocatable' | grep nvidia.com/gpu
  ```

---

## 5. Setting up Istio Service Mesh

### **Creating Istio Namespace**
```bash
kubectl create namespace istio-system
```

### **Installing Istio Base Components**
```bash
helm install istio-base istio/base \
  --version 1.20.2 \
  --namespace istio-system --wait
```

### **Deploying Istio Control Plane (Istiod)**
```bash
helm install istiod istio/istiod \
  --version 1.20.2 \
  --namespace istio-system --wait
```
- Verify installation with:
  ```bash
  kubectl get pods -n istio-system
  ```

---

## Conclusion

By following these steps, you have successfully:
- Set up an **Amazon EKS cluster**
- Configured **IAM roles** and **service accounts**
- Installed **Istio** for service mesh
- Deployed **AWS Load Balancer Controller**
- Installed **KServe for model serving**
- Configured **NVIDIA GPU Operator** for machine learning workloads

For troubleshooting, use:
```bash
kubectl logs <pod-name> -n <namespace>
```

pods: 
NAMESPACE       NAME                                                                  READY   STATUS      RESTARTS      AGE
cert-manager    pod/cert-manager-57d855897b-tdf9z                                     1/1     Running     0             46m
cert-manager    pod/cert-manager-cainjector-5c7f79b84b-x8wdj                          1/1     Running     0             46m
cert-manager    pod/cert-manager-webhook-657b9f664c-brqp4                             1/1     Running     0             46m
default         pod/kserve-controller-manager-6cb87dcc55-vnr9z                        3/3     Running     0             40m
default         pod/modelmesh-controller-6f5bdb97db-mlxmg                             2/2     Running     1 (40m ago)   40m
default         pod/torchserve-sd3-predictor-855444854-qfcbj                          2/2     Running     0             37m
gpu-operator    pod/gpu-feature-discovery-br28q                                       1/1     Running     0             49m
gpu-operator    pod/gpu-operator-1737987260-node-feature-discovery-gc-75fd956f2pj4w   1/1     Running     0             49m
gpu-operator    pod/gpu-operator-1737987260-node-feature-discovery-master-54cc6j896   1/1     Running     0             49m
gpu-operator    pod/gpu-operator-1737987260-node-feature-discovery-worker-c8xnx       1/1     Running     0             49m
gpu-operator    pod/gpu-operator-bcf5659c7-xp58g                                      1/1     Running     0             49m
gpu-operator    pod/nvidia-container-toolkit-daemonset-b8j5z                          1/1     Running     0             49m
gpu-operator    pod/nvidia-cuda-validator-h44m6                                       0/1     Completed   0             49m
gpu-operator    pod/nvidia-dcgm-exporter-7hs6z                                        1/1     Running     0             49m
gpu-operator    pod/nvidia-device-plugin-daemonset-66mmv                              1/1     Running     0             49m
gpu-operator    pod/nvidia-operator-validator-txscc                                   1/1     Running     0             49m
istio-ingress   pod/istio-ingress-8689988bc9-znwm4                                    1/1     Running     0             47m
istio-system    pod/istiod-7dbdb8d5bd-rmtbc                                           1/1     Running     0             48m
kube-system     pod/aws-load-balancer-controller-7bf7b6d7f-gck7t                      1/1     Running     0             38m
kube-system     pod/aws-load-balancer-controller-7bf7b6d7f-gr6rn                      1/1     Running     0             38m
kube-system     pod/aws-node-xqhq8                                                    2/2     Running     0             101m
kube-system     pod/coredns-6c55b85fbb-nsvwb                                          1/1     Running     0             104m
kube-system     pod/coredns-6c55b85fbb-wkmnj                                          1/1     Running     0             104m
kube-system     pod/kube-proxy-kr6rp                                                  1/1     Running     0             101m
kube-system     pod/metrics-server-d5865ff47-mbg7v                                    1/1     Running     0             46m

NAMESPACE       NAME                                        TYPE           CLUSTER-IP       EXTERNAL-IP                                                                      PORT(S)                                      AGE
cert-manager    service/cert-manager                        ClusterIP      10.100.45.201    <none>                                                                           9402/TCP                                     46m
cert-manager    service/cert-manager-cainjector             ClusterIP      10.100.246.40    <none>                                                                           9402/TCP                                     46m
cert-manager    service/cert-manager-webhook                ClusterIP      10.100.50.199    <none>                                                                           443/TCP,9402/TCP                             46m
default         service/kserve-controller-manager-service   ClusterIP      10.100.199.191   <none>                                                                           8443/TCP                                     40m
default         service/kserve-webhook-server-service       ClusterIP      10.100.241.203   <none>                                                                           443/TCP                                      40m
default         service/kubernetes                          ClusterIP      10.100.0.1       <none>                                                                           443/TCP                                      107m
default         service/modelmesh-serving                   ClusterIP      None             <none>                                                                           8033/TCP,8008/TCP,2112/TCP                   40m
default         service/modelmesh-webhook-server-service    ClusterIP      10.100.139.157   <none>                                                                           9443/TCP                                     40m
default         service/torchserve-sd3-predictor            ClusterIP      10.100.97.88     <none>                                                                           80/TCP                                       37m
gpu-operator    service/gpu-operator                        ClusterIP      10.100.134.10    <none>                                                                           8080/TCP                                     49m
gpu-operator    service/nvidia-dcgm-exporter                ClusterIP      10.100.67.11     <none>                                                                           9400/TCP                                     49m
istio-ingress   service/istio-ingress                       LoadBalancer   10.100.189.131   k8s-istioing-istioing-8fab4b3d41-ff52dca464b77c08.elb.ap-south-1.amazonaws.com   15021:32489/TCP,80:31052/TCP,443:32522/TCP   47m
istio-system    service/istiod                              ClusterIP      10.100.190.50    <none>                                                                           15010/TCP,15012/TCP,443/TCP,15014/TCP        48m
kube-system     service/aws-load-balancer-webhook-service   ClusterIP      10.100.110.233   <none>                                                                           443/TCP                                      38m
kube-system     service/eks-extension-metrics-api           ClusterIP      10.100.22.61     <none>                                                                           443/TCP                                      107m
kube-system     service/kube-dns                            ClusterIP      10.100.0.10      <none>                                                                           53/UDP,53/TCP,9153/TCP                       104m
kube-system     service/metrics-server                      ClusterIP      10.100.66.220    <none>                                                                           443/TCP                                      46m

NAMESPACE      NAME                                                                   DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR                                                          AGE
gpu-operator   daemonset.apps/gpu-feature-discovery                                   1         1         1       1            1           nvidia.com/gpu.deploy.gpu-feature-discovery=true                       49m
gpu-operator   daemonset.apps/gpu-operator-1737987260-node-feature-discovery-worker   1         1         1       1            1           <none>                                                                 49m
gpu-operator   daemonset.apps/nvidia-container-toolkit-daemonset                      1         1         1       1            1           nvidia.com/gpu.deploy.container-toolkit=true                           49m
gpu-operator   daemonset.apps/nvidia-dcgm-exporter                                    1         1         1       1            1           nvidia.com/gpu.deploy.dcgm-exporter=true                               49m
gpu-operator   daemonset.apps/nvidia-device-plugin-daemonset                          1         1         1       1            1           nvidia.com/gpu.deploy.device-plugin=true                               49m
gpu-operator   daemonset.apps/nvidia-device-plugin-mps-control-daemon                 0         0         0       0            0           nvidia.com/gpu.deploy.device-plugin=true,nvidia.com/mps.capable=true   49m
gpu-operator   daemonset.apps/nvidia-driver-daemonset                                 0         0         0       0            0           nvidia.com/gpu.deploy.driver=true                                      49m
gpu-operator   daemonset.apps/nvidia-mig-manager                                      0         0         0       0            0           nvidia.com/gpu.deploy.mig-manager=true                                 49m
gpu-operator   daemonset.apps/nvidia-operator-validator                               1         1         1       1            1           nvidia.com/gpu.deploy.operator-validator=true                          49m
kube-system    daemonset.apps/aws-node                                                1         1         1       1            1           <none>                                                                 104m
kube-system    daemonset.apps/kube-proxy                                              1         1         1       1            1           <none>                                                                 104m

NAMESPACE       NAME                                                                    READY   UP-TO-DATE   AVAILABLE   AGE
cert-manager    deployment.apps/cert-manager                                            1/1     1            1           46m
cert-manager    deployment.apps/cert-manager-cainjector                                 1/1     1            1           46m
cert-manager    deployment.apps/cert-manager-webhook                                    1/1     1            1           46m
default         deployment.apps/kserve-controller-manager                               1/1     1            1           40m
default         deployment.apps/modelmesh-controller                                    1/1     1            1           40m
default         deployment.apps/modelmesh-serving-mlserver-1.x                          0/0     0            0           40m
default         deployment.apps/modelmesh-serving-ovms-1.x                              0/0     0            0           40m
default         deployment.apps/modelmesh-serving-torchserve-0.x                        0/0     0            0           40m
default         deployment.apps/modelmesh-serving-triton-2.x                            0/0     0            0           40m
default         deployment.apps/torchserve-sd3-predictor                                1/1     1            1           37m
gpu-operator    deployment.apps/gpu-operator                                            1/1     1            1           49m
gpu-operator    deployment.apps/gpu-operator-1737987260-node-feature-discovery-gc       1/1     1            1           49m
gpu-operator    deployment.apps/gpu-operator-1737987260-node-feature-discovery-master   1/1     1            1           49m
istio-ingress   deployment.apps/istio-ingress                                           1/1     1            1           47m
istio-system    deployment.apps/istiod                                                  1/1     1            1           48m
kube-system     deployment.apps/aws-load-balancer-controller                            2/2     2            2           38m
kube-system     deployment.apps/coredns                                                 2/2     2            2           104m
kube-system     deployment.apps/metrics-server                                          1/1     1            1           46m

NAMESPACE       NAME                                                                               DESIRED   CURRENT   READY   AGE
cert-manager    replicaset.apps/cert-manager-57d855897b                                            1         1         1       46m
cert-manager    replicaset.apps/cert-manager-cainjector-5c7f79b84b                                 1         1         1       46m
cert-manager    replicaset.apps/cert-manager-webhook-657b9f664c                                    1         1         1       46m
default         replicaset.apps/kserve-controller-manager-6cb87dcc55                               1         1         1       40m
default         replicaset.apps/modelmesh-controller-6f5bdb97db                                    1         1         1       40m
default         replicaset.apps/modelmesh-serving-mlserver-1.x-57d65d9fdd                          0         0         0       40m
default         replicaset.apps/modelmesh-serving-ovms-1.x-5488c8f4f9                              0         0         0       40m
default         replicaset.apps/modelmesh-serving-torchserve-0.x-67f9485cb9                        0         0         0       40m
default         replicaset.apps/modelmesh-serving-triton-2.x-66756bc646                            0         0         0       40m
default         replicaset.apps/torchserve-sd3-predictor-855444854                                 1         1         1       37m
gpu-operator    replicaset.apps/gpu-operator-1737987260-node-feature-discovery-gc-75fd956f4c       1         1         1       49m
gpu-operator    replicaset.apps/gpu-operator-1737987260-node-feature-discovery-master-54cccf99dd   1         1         1       49m
gpu-operator    replicaset.apps/gpu-operator-bcf5659c7                                             1         1         1       49m
istio-ingress   replicaset.apps/istio-ingress-8689988bc9                                           1         1         1       47m
istio-ingress   replicaset.apps/istio-ingress-94f46b75b                                            0         0         0       47m
istio-system    replicaset.apps/istiod-7dbdb8d5bd                                                  1         1         1       48m
kube-system     replicaset.apps/aws-load-balancer-controller-7bf7b6d7f                             2         2         2       38m
kube-system     replicaset.apps/coredns-6c55b85fbb                                                 2         2         2       104m
kube-system     replicaset.apps/metrics-server-d5865ff47                                           1         1         1       46m

NAMESPACE       NAME                                                           REFERENCE                             TARGETS       MINPODS   MAXPODS   REPLICAS   AGE
default         horizontalpodautoscaler.autoscaling/torchserve-sd3-predictor   Deployment/torchserve-sd3-predictor   cpu: 0%/80%   1         1         1          37m
istio-ingress   horizontalpodautoscaler.autoscaling/istio-ingress              Deployment/istio-ingress              cpu: 2%/80%   1         5         1          47m
istio-system    horizontalpodautoscaler.autoscaling/istiod                     Deployment/istiod                     cpu: 0%/80%   1         5         1          48m


Kiali Graph : 

<img width="1781" alt="image" src="https://github.com/user-attachments/assets/ef45d84b-db60-43d0-8f03-499440222447" />
<img width="1592" alt="image" src="https://github.com/user-attachments/assets/e73e30f8-2c1b-4e8a-90ab-f77f8e1c24b7" />
<img width="1792" alt="image" src="https://github.com/user-attachments/assets/247c1d36-a392-4bba-9cd8-4e148b4ca713" />
<img width="1582" alt="image" src="https://github.com/user-attachments/assets/d50ff13a-6279-4ad8-a33c-4f666e33c59e" />
<img width="1785" alt="image" src="https://github.com/user-attachments/assets/19159f67-ce24-455a-a27f-6983d12fee3a" />

Grafana Logs : 

<img width="1780" alt="image" src="https://github.com/user-attachments/assets/1e31e891-de8c-4a0b-bb81-10e7f60a79ac" />
<img width="1792" alt="image" src="https://github.com/user-attachments/assets/9dd3d702-17dc-439a-b17a-5a5597279b5f" />
<img width="1791" alt="image" src="https://github.com/user-attachments/assets/faf0bc08-994d-4abe-a1d3-a47b0e03f7f3" />
<img width="1768" alt="image" src="https://github.com/user-attachments/assets/df359aa3-86c6-44da-9f16-9125b67cf19d" />
<img width="1792" alt="image" src="https://github.com/user-attachments/assets/76df4183-939d-4db0-afec-1cd3ed79b761" />
<img width="1791" alt="image" src="https://github.com/user-attachments/assets/addb5220-ecb0-4eae-b441-d09cf4dcfe49" />



output-image:





