# Deploy on Kubernetes

KubeRay is a prerequisite for the deployment.

## Build KubeRay cluster image

Generate credentials in image for AWS to download models from AWS S3 storage with the command `aws s3 cp`.

Add `/home/ray/.aws/config` and `/home/ray/.aws/credentials` to image as below.

```
$ cat /home/ray/.aws/config
[default]

$ cat /home/ray/.aws/credentials
[default]
aws_access_key_id = dlwuVbseIlXR5GppudjxH
aws_secret_access_key = aNINkp4MwMtthZpjbflxlsf3uZUaNpmBSQ0dcubgJ
```

Build the base image of KubeRay with the docker file `deploy/ray/Dockerfile-base`. The default base image is `opencsg/llm-inference:base-0.0.3`.

```
./build_llmserve_image_base.sh
```

Build KubeRay image with docker file `deploy/ray/Dockerfile`. The default image is `opencsg/llm-inference:0.1.0-<COMMIT ID>`.

```
./build_llmserve_image.sh
```

## Deploy KubeRay operator on Kubernetes

Install KubeRay operator online:

```
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Confirm the repo exists
helm search repo kuberay --devel

# Install both CRDs and KubeRay operator v1.1.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0
```

Install KubeRay operator with local tgz file:

```
helm install --namespace ray kuberay-operator ./kuberay-operator-1.1.0.tgz
```

Install KubeRay operator from local directory:

```
cd kuberay-1.0.0/helm-chart/kuberay-operator
helm install kuberay-operator --namespace ray .
helm uninstall kuberay-operator --namespace ray
```

See reference [here](https://github.com/ray-project/kuberay)

## Enable GPUs on Kubernetes

Prior to installing the GPU plugin, enable GPUs of container runtime.
See reference [here](https://github.com/NVIDIA/k8s-device-plugin).

```
wget https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.3/nvidia-device-plugin.yml
kubectl apply -f nvidia-device-plugin.yml
```

After running the command `kubectl describe node`, the GPU resource `nvidia.com/gpu` will be displayed on the node.

```
kubectl describe node xxxx-gpu1-4090


Capacity:
  cpu:                32
  ephemeral-storage:  488510864Ki
  hugepages-1Gi:      0
  hugepages-2Mi:      0
  memory:             131728152Ki
  nvidia.com/gpu:     2
  pods:               110
```

## Create a Ray cluster with build image

Update image `sean/llmray:0.0.2-<COMMIT ID>` in `deploy/kuberay/starray.yaml`

Update volume mount config in `deploy/kuberay/starray.yaml` for models.

```
# create namespace
kubectl create ns ray

# create cluster
kubectl apply -f deploy/kuberay/starray.yaml
```

## Test GPT2 model by running command in the pod

If the model is saved on AWS storage, make sure to successfully run `aws s3 cp` to download the model files.

```
llm-serve start serving-ui --model=/home/ray/models/text-generation--gpt2.yaml
```

## Access model

Export port `8000` through Nginx on Kubernetes cluster and then access the model `http://<IP>:8000/gpt2`.

## Clean up Kubernetes resources

Delete Ray cluster:

```
kubectl delete -f deploy/kuberay/starray.yaml
```

Uninstall the KubeRay operator chart:

```
helm uninstall kuberay-operator
```
