# LLM Inference - Quickly Deploy Productive LLM Service

[中文文档](./README_cn.md)

`LLM Inference` is a large language model serving solution for deploying productive LLM services.

We gained a great deal of inspiration and motivation from [this open source project](https://github.com/ray-project/ray-llm). We are incredibly grateful to them for providing us with the chance to further explore and innovate by standing on the shoulders of giants.

<img src="./docs/llm-inference.png" alt="image" width=600 height="auto">

Llm-inference is a platform for deploying and managing LLM (Lifelong Learning Machine) inference tasks with the following features:

- Utilizes Ray technology to organize multiple nodes into a cluster, achieving centralized management of computational resources and distributing resources required for each inference task.
- Provides a comprehensive management interface to monitor various states of LLM inference tasks, including resource utilization, the number of replicas, logs, etc.
- Supports automatic scaling out of inference tasks, dynamically adjusting computational resources based on the volume of requests to meet user needs at different times and optimizing resource usage.
- Implements serverless inference by automatically shutting down resources when there are no active inference tasks, preventing unnecessary resource waste.
- Supports various inference frameworks and formats, including hg transformer (PyTorch), DeepSpeed, GGUF, VLLM, etc., with an ongoing expansion of supported frameworks.
- Establishes user-friendly inference task publishing standards using YAML configurations for model inference loading and execution parameters, such as the framework used, batch size, serverless scaling policies, and more, to lower the barrier to entry for users.
- Provides REST API or User Interface (UI) support, facilitating access to and management of model inference tasks.
- Enables streaming capabilities.
- Supports multiple methods for retrieving models, including from OpenCSG Model Hub, Huggingface Hub, or through customized S3 storage and local storage solutions.

More features in [Roadmap](./Roadmap.md) are coming soon.

## Getting started

### Deploy locally

#### Install `LLM Inference` and dependencies

You can start by cloning the repository and pip install `llm-serve`. It is recommended to deploy `llm-serve` with Python 3.10+.

```
git clone https://github.com/OpenCSGs/llm-inference.git
cd llm-inference
pip install .
```

Option to use another pip source for faster transfer if needed.

```
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

Install specified dependencies by components:

```
pip install '.[backend]'
pip install '.[frontend]'
```

**Note:** Install vllm dependency if runtime supports GPUs, run the following command:

```
pip install '.[vllm]'
```

Option to use other pip sources for faster transfers if needed.

```
pip install '.[backend]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install '.[frontend]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install '.[vllm]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### Install Ray and start a Ray Cluster locally

Pip install Ray:

```
pip install -U "ray[serve-grpc]==2.9.3"
```

Option to use another pip source for faster transfer if needed.

```
pip install -U "ray[serve-grpc]==2.9.3" -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

> **Note:** ChatGLM2-6b requires transformers<=4.33.3, while the latest vllm requires transformers>=4.36.0.

Start cluster then:

```
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

See reference [here](https://docs.ray.io/en/releases-2.9.3/ray-overview/installation.html).

#### Quick start

You can follow the [quick start](./docs/quick_start.md) to run an end-to-end case for model serving.

#### Uninstall

Uninstall `llm-serve` package:

```
pip uninstall llm-serve
```

Then shutdown the `Ray` cluster:

```
ray stop
```

### API server

See the [guide](./docs/api_server.md) for API server and API documents.

### Deploy on bare metal

See the [guide](./docs/deploy_on_bare_metal.md) to deploy on bare metal.

### Deploy on kubernetes

See the [guide](./docs/deploy_on_kubernetes.md) to deploy on kubernetes.

## FAQ

### How to use model from local path or git server or S3 storage or OpenCSG Hub

See the [guide](./docs/git_server_s3_storage.md) for how to use model from local path or git server or S3 storage.

### How to add new models using LLMServe Model Registry

LLMServe allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, see the [LLMServe Model Registry](./models/README.md).

### Developer Guide

See the [Developer Guide](./docs/developer.md) for how to setup a development environment so you can get started contributing.

### Common Issues

See the [document](./docs/common_issues.md) for some common issues.
