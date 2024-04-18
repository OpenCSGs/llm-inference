# LLM Inference - 快速部署生产级的LLM服务

`LLM Inference` 是一个用于部署企业生产级LLM服务和推理的解决方案。

我们从[这个开源项目](https://github.com/ray-project/ray-llm)中获得了大量的灵感和动力。非常感谢此项目为我们提供了站在巨人的肩膀上进一步探索和创新的机会。

<img src="./docs/llm-inference.png" alt="image" width=600 height="auto">

Llm-inference 是一个用于部署和管理LLM（Lifelong Learning Machine）推理任务的平台，它提供以下特点：
- 利用 Ray 技术将多个节点组织成一个集群，实现统一计算资源的管理，并对每个推理任务所需的资源进行分配。
- 提供全方位的管理界面，以监控LLM推理任务的各种状态，包括资源使用情况、副本数量、日志记录等。
- 支持推理任务的自动扩展（Scale Out），根据请求量自动调整计算资源，以适应各个时间段的用户需求，并优化资源使用。
- 实现无服务器（Serverless）推理，在没有推理任务执行时自动关闭相关资源，防止不必要的资源浪费。
- 支持多种推理框架和格式，包括 hg transformer （PyTorch）、DeepSpeed、GGUF、VLLM 等，并持续扩充支持的框架列表。
- 制定了易于理解的推理任务发布规范，通过 YAML 格式配置模型推理的加载和执行参数，例如所使用的框架、批处理大小、无服务器扩缩容策略等，以降低用户的使用门槛。
- 提供REST API或用户界面（UI）支持，方便模型推理任务的访问和管理。
- 支持流式处理（Streaming）功能。
- 支持多种模型获取方式，包括从 OpenCSG Model Hub、Huggingface Hub 获取，或通过自定义S3存储和本地存储解决方案。

更多[开发路线图](./Roadmap.md)中的功能正在开发中，欢迎您的贡献。

## 本地部署

### 部署`LLM Inference`及其依赖

您可以下载此项目代码，然后使用pip install ' llm-serve '安装。建议使用Python 3.10+部署'llm-serve '。

```
git clone https://github.com/OpenCSGs/llm-inference.git
cd llm-inference
```

按组件安装指定的依赖项:

```
pip install '.[backend]'
```

**注意:** vllm是可选的，因为它需要GPU, 根据你环境的情况安装:

```
pip install '.[vllm]'
```

安装 `llm-inference`:
```
pip install .
```

如果您受到网络传输速度的限制或影响，可以选择使用更快的传输速度的pip源。

```
pip install '.[backend]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install '.[vllm]' -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 安装Ray并在本地启动Ray Cluster

启动Ray集群:

```
ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265
```

#### 快速入门

你可以按照[快速入门](./docs/quick_start.md)来运行一个端到端的模型服务案例。

## 其他事项

### 使用模型从本地路径或git服务器或S3存储或OpenCSG Model Hub

请参见[此文档](./docs/git_server_s3_storage.md)的内容，查看如何从本地路径或git服务器或S3存储使用模型。

### 使用LLMServe模型注册表添加新模型

LLMServe允许您通过添加单个配置文件轻松添加新模型。 以了解有关如何自定义或添加新模型的更多信息，请参阅[此文档](./models/README.md)。

### 开发者指南

在贡献之前，请参阅[开发者指南](./docs/developer.md).

### 常见问题

更多问题，请参阅[此文档](./docs/common_issues.md)。
