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

