# Quick start in the command line interface

## Introduction to llm-serve

`llmserve` comes with its own CLI, `llm-serve`, which allows you to interact directly with the backend.

Installing `llmserve` also installs the `llm-serve` CLI, and you can get a list of all available commands by running `llm-serve --help`.

```SHELL
# llm-serve --help

 Usage: llm-serve [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ list         List available model(s) and deployed serving etc.                                                                                                     │
│ predict      Predict one or several models with one or multiple prompts, optionally read from file, and save the results to a file.                                │
│ start        Start application(s) for LLM serving, API server, experimention, fine-tuning and comparation.                                                         │
│ stop         Stop application(s) for LLM serving and API server.                                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Start a model serving

You can deploy any model in the [models](../models) directory of this repo, or define your own model YAML file and run that instead.  
For example:

```
llm-serve start serving-rest --model models/text-generation--facebook--opt-125m.yaml
```

## Check model serving status and predict URL

Check model serving status and predict URL by:

```SHELL
# llm-serve list serving --appname default
{
  "default": {
    "status": {
      "default": {
        "application_status": "RUNNING",
        "deployments_status": {
          "facebook--opt-125m": "HEALTHY",
          "facebook--opt-125m-router": "HEALTHY"
        }
      }
    },
    "url": {
      "facebook/opt-125m": "http://0.0.0.0:8000/api/v1/default/facebook--opt-125m/run/predict"
    }
  }
}
```

## invoke model serving

Invoke model with command `llm-serve predict`

```
llm-serve predict --model facebook/opt-125m --prompt "I am going to do"
```

If you start the model using `llm-serve start serving-rest`, you can also run the following command `curl` to call the model predict API shown above.

```
curl -H "Content-Type: application/json" -X POST -d '{"prompt": "What can I do"}' "http://127.0.0.1:8000/api/v1/default/facebook--opt-125m/run/predict"

curl -H "Content-Type: application/json" -X POST -d '{"prompt": ["What can I do", "How are you"]}' "http://127.0.0.1:8000/default-d2b9814399fd/facebook--opt-125m/run/predict"
```

Run the following command `curl` to call the model predict API will return data in OpenAI style.

```
curl -H "Content-Type: application/json" -X POST -d '{"stream": false, "messages": [{"role": "user", "content": "Say this is a test!"}]}' "http://127.0.0.1:8000/api/v1/default/facebook--opt-125m/chat/completions"
```

Run the following python code to call model predict API with stream output.

```
jsonData = {"stream": True, "messages": [{"role": "user", "content": "Say this is a test!"}]}
url = "http://127.0.0.1:8000/api/v1/default/facebook--opt-125m/chat/completions"

response = requests.post(url=url, json=jsonData, stream=True)
response.raise_for_status()
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="")
```

## Start a model serving with Gradio UI

You can start a trial with the following command, which will start a serving and built-in UI for the model running on <http://127.0.0.1:8000/facebook--opt-125m>.

```
llm-serve start serving-ui --model=models/text-generation--facebook--opt-125m.yaml
```

You can enjoy the trial with a simple `Gradio` UI.

## Stop model serving

Run command `llm-serve stop serving` to delete model serving.

```
llm-serve stop serving --appname facebook--opt-125m
```
