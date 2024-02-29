# To-do list

- Automatically upgrades models when a new version is available on the model hub
- Support for comparing multiple models via API
- Support for image/video models
- Support for embedding models
- Multi-round of chat
- Provide scripts for deployment/uninstallation process
- Integration with CSGHub API to download models

# Preliminary thoughts on LLM Inference roadmap

1. Estimate the inference speed and throughput of LLM, or the number of users that can be accommodated with acceptance delay. With the feature, user can know how many GPUs are required for specifed the number of concurrent users, or how many concurrent requests the env (GPUs) can handle.
2. The LLM instance is automatically invoked to scale up if the number of parallel users increases, otherwise to scale down even to zero if no user to free up GPU resource.
3. Upgrade or downgrade Canary to a new or other LLM in inference.
4. Pre and Post actions for LLM (user-define supported), e.g. check if there are valid outputs?
5. Support for Kafka to handle large throughputs/requests.
6. Inference observability and model inference monitoring.
7. Muti-LLMs inference work together (related toAI Agents?)
8. Inference and application workflow diagrams?
9. Build default application like RAG for fastest LLM inference, vLLM + Ray, increases throughputs and reduces GPU resources.
