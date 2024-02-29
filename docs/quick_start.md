# Quick start in the command line interface

## Introduction to llm-serve

`llmserve` comes with its own CLI, `llm-serve`, which allows you to interact directly with the backend without having to use the Gradio frontend.

Installing `llmserve` also installs the `llm-serve` CLI, and you can get a list of all available commands by running `llm-serve --help`.

```SHELL
# llm-serve --help

 Usage: llm-serve [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ evaluate     Evaluate and summarize the results of a multi_query run with a strong 'evaluator' LLM like GPT-4.                                                     │
│ list         List available model(s) and deployed serving etc.                                                                                                     │
│ predict      Predict one or several models with one or multiple prompts, optionally read from file, and save the results to a file.                                │
│ start        Start application(s) for LLM serving, API server, experimention, fine-tuning and comparation.                                                         │
│ stop         Stop application(s) for LLM serving and API server.                                                                                                   │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Start a model serving

You can deploy any model in the `models` directory of this repo, or define your own model YAML file and run that instead.  
For example:

```
llm-serve start serving --model=models/text-generation--gpt2.yaml

# You can start mutiple models serving at once.
llm-serve start serving --model=models/text-generation--facebook--opt-125m.yaml --model=models/text-generation--gpt2.yaml
```

## Check model serving status and predict URL

Check model serving status and predict URL by:

```SHELL
# llm-serve list serving --name gpt2
{
  "gpt2": {
    "status": {
      "gpt2": {
        "application_status": "RUNNING",
        "deployments_status": {
          "gpt2": "HEALTHY",
          "RouterDeployment": "HEALTHY"
        }
      }
    },
    "url": {
      "prodict_url": "http://0.0.0.0:8000/api/v1/default/gpt2/run/predict"
    }
  }
}
```

## Using the model serving

Invoke model with command `llm-serve predict`

```
llm-serve predict --model gpt2 --prompt "I am going to do" --prompt "What do you like" 
```

If you start the model using `llm-serve start serving`, you can also run the following command `curl` to call the model predict API shown above.

```
curl -H "Content-Type: application/json" -X POST -d '{"prompt": "What can I do"}' "http://127.0.0.1:8000/api/v1/default/gpt2/run/predict"

curl -H "Content-Type: application/json" -X POST -d '[{"prompt":"How can you"}, {"prompt": "What can I do"}]' "http://127.0.0.1:8000/api/v1/default/gpt2/run/predict"
```

## Start your trial

You can start a trial with the following command, which will start a serving and built-in UI for the model running on <http://127.0.0.1:8000/facebook--opt-125m>.

```
llm-serve start experimental --model=models/text-generation--facebook--opt-125m.yaml
```

You can enjoy the trial with a simple `Gradio` UI.

## Stop model serving

Run command `llm-serve stop serving` to delete model serving.

```
llm-serve stop serving --appname facebook--opt-125m
```
