# Deploy OpenCSG LLM Inference on Bare Metal

Ray cluster can be deploy on physical or virtual machines.

Bootstrap Ray cluster for VM on bare metal depends on container runtime and SSH access between all hosts on the ray cluster (no password required). SSH private key defaults to `~/.ssh/id_rsa`.

Install Ray on bootstrap hosts and update `deploy/ray/llmserve-cluster.yaml` with correct values, such as `image`, `head_ip`, `worker_ips`, or other values.

Start Ray cluster:

```
ray up --no-config-cache deploy/ray/llmserve-cluster.yaml
```

Stop Ray cluster:

```
ray down deploy/ray/llmserve-cluster.yaml
```

See reference [here](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html#on-prem).
