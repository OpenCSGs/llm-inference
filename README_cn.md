# LLM Inference - 快速部署生产级的LLM服务

`LLM Inference` 是一个用于部署企业生产级LLM服务和推理的解决方案。

我们从[这个开源项目](https://github.com/ray-project/ray-llm)中获得了大量的灵感和动力。非常感谢此项目为我们提供了站在巨人的肩膀上进一步探索和创新的机会。

<img src="./docs/llm-inference.png" alt="image" width=600 height="auto">

使用此解决方案，您可以: 
 
- 在CPU/GPU上快速部署各种LLM。 
- 通过Ray集群在多个节点上部署LLM。 
- 使用vLLM引擎构建LLM推理，加快推理速度。 
- 利用Restful API管理模型推理。 
- 使用YAML自定义模型部署。 
- 比较模型推断。

更多[开发路线图](./Roadmap.md)中的功能正在开发中，欢迎您的贡献。

## 快速入门

### 本地部署

#### 部署`LLM Inference`及其依赖

您可以下载此项目代码，然后使用pip install ' llm-serve '安装。建议使用Python 3.10+部署'llm-serve '。

```
git clone https://github.com/OpenCSGs/llm-inference.git
cd llm-inference
pip install .
```

如果您受到网络传输速度的限制或影响，可以选择使用更快的传输速度的pip源。

```
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

按组件安装指定的依赖项:

```
pip install '.[backend]'
pip install '.[frontend]'
```

**注意:** 如果运行时支持gpu，请执行以下命令安装vllm依赖:

```
pip install '.[vllm]'
```

如果您受到网络传输速度的限制或影响，可以选择使用更快的传输速度的pip源。

```
pip install '.[backend]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install '.[frontend]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install '.[vllm]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 安装Ray并在本地启动Ray Cluster

安装Ray:

```
pip install -U "ray[serve-grpc]==2.8.0"
```

如果您受到网络传输速度的限制或影响，可以选择使用更快的传输速度的pip源。

```
pip install -U "ray[serve-grpc]==2.8.0" -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

> **注意:** ChatGLM2-6b要求transformers<=4.33.3，最新的vllm要求transformers>=4.36.0。

启动Ray集群:

```
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

请参阅[此文档](https://docs.ray.io/en/releases-2.8.0/ray-overview/installation.html)获取更多Ray的安装与启动的信息.

#### 快速入门

你可以按照[快速入门](./docs/quick_start.md)来运行一个端到端的模型服务案例。

#### 卸载

卸载 `llm-serve` :

```
pip uninstall llm-serve
```

停止`Ray` 集群:

```
ray stop
```

### API服务器

关于详细的API Server和API的内容，请参见[此文档](./docs/api_server.md)。

### 在裸机上部署

请参见[此文档](./docs/deploy_on_bare_metal.md)中描述，查阅如何在裸机上部署。

### 在Kubernetes中部署

请参见[此文档](./docs/deploy_on_kubernetes.md)，查阅如何在Kubernetes中部署。

## 其他事项

### 使用模型从本地路径或git服务器或S3存储或OpenCSG Model Hub

请参见[此文档](./docs/git_server_s3_storage.md)的内容，查看如何从本地路径或git服务器或S3存储使用模型。

### 使用LLMServe模型注册表添加新模型

LLMServe允许您通过添加单个配置文件轻松添加新模型。 以了解有关如何自定义或添加新模型的更多信息，请参阅[此文档](./models/README.md)。

### 开发者指南

在贡献之前，请参阅[开发者指南](./docs/developer.md).

### 常见问题

更多问题，请参阅[此文档](./docs/common_issues.md)。
