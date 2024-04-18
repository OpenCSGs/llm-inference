# LLM Inference - Quickly Deploy Productive LLM Service

[中文文档](./README_cn.md)

`LLM Inference` is a large language model serving solution for deploying productive LLM services.

We gained a great deal of inspiration and motivation from [this open source project](https://github.com/ray-project/ray-llm). We are incredibly grateful to them for providing us with the chance to further explore and innovate by standing on the shoulders of giants.

<img src="./docs/llm-inference.png" alt="image" width=600 height="auto">

### TL;DR

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


## Deployment

### Install `LLM Inference` and dependencies

You can start by cloning the repository and pip install `llm-serve`. It is recommended to deploy `llm-serve` with Python 3.10+.

```
git clone https://github.com/OpenCSGs/llm-inference.git
cd llm-inference
```

Install specified dependencies by components:

```
pip install '.[backend]'
```

**Note:** `vllm` is optional, since it requires GPU:

```
pip install '.[vllm]'
```

Install `llm-inference`:
```
pip install .
```

### Start a Ray Cluster locally

```
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

### Quick start

You can follow the [quick start](./docs/quick_start.md) to run an end-to-end case.


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
