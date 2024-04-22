# How to use model from local path or git server or S3 storage

## Pre-requirement

Run command `aws configure` to set `Access key` and `Secrete key` for S3 storage.

## Configure model with custom storage

Enable `endpoint_url` in model.yaml file to point to custom S3 storage endpoint URL.

```
# from AWS S3
  initialization:
    s3_mirror_config: 
      # bucket_uri: s3://gpt2/facemodel/  # Must include hash file with commit id in the repo

# from custom S3 storage such as minio
  initialization:
    s3_mirror_config:
      endpoint_url: http://3.107.108.170:9000 # for custom S3 storage endpoint url 
      bucket_uri: s3://gpt2/facemodel/  # Must include hash file with commit id in the repo

# from local path
  initialization:
    s3_mirror_config:
      bucket_uri: /Users/hub/models/gpt2/ # Local path of model with hash file

# from git server
  initialization:
    s3_mirror_config:
      git_uri: https://portal.opencsg.com/models/AIWizards/gpt2.git # git address for git clone
```

To use the local path of the model, set `bucket_uri` to the local path. For example: `bucket_uri: /Users/home/models/gpt2/`.

Make sure there is a file `hash` include <commit-id> under model path as below.

```
llm-inference user$ cat /Users/models/opt-125m/hash
27dcfa74d334bc871f3234de431e71c6eeba5dd6
```

To download the model using `git clone`, set `git_uri` to the model address.

# How to auto download model from OpenCSG Hub

- Add `runtime_env` section to model yaml config file.
- Update model_id according to [OpenCSG Model Hub](https://portal.opencsg.com/models).

For example:
```
  initialization:
    runtime_env:
      env_vars:
        HF_ENDPOINT: https://hub.opencsg.com/hf
```