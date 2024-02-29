# Common Issues

`AttributeError: 'ChatGLMTokenizer' object has no attribute 'tokenizer'`

Transformer version should be 4.33.3.

## Launch model by Ray Job API

```
import requests
import json
import time

entry_point = "/home/ray/anaconda3/bin/python /home/ray/anaconda3/bin/llm-serve run-experimental --model models/amazon--LightGPT.yaml"
resp = requests.post(
    "http://127.0.0.1:8265/api/jobs/",
    json={
        "entrypoint": entry_point,
        "runtime_env": {},
        "job_id": None,
        "metadata": {"job_submission_id": "123"}
    }
)
rst = json.loads(resp.text)
submission_id = rst["job_id"]
print(submission_id)
```

See the [API documents](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/api.html#/paths/~1api~1jobs/post) for more details.
