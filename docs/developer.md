# Developer Guide

This doc explains how to setup a development environment so you can get started contributing.

## Prerequisites

Follow the instructions below to set up your development environment. 

### Install requirements

You must install these tools:

1. [`Python`](https://www.python.org/downloads/): The project needs Python 3.10+ for development.
1. [`Ray`](https://docs.ray.io/en/master/ray-overview/installation.html): The project depends on Ray cluster.

### Setup your environment

Create Python virtual environments.
```
$ python -m venv ./venv
```

Activate the virtual environment
```
$ source venv/bin/activate
```

### Checkout your fork

To check out this repository:

1. Create your own
   [fork of this repo](https://help.github.com/articles/fork-a-repo/)
1. Clone it to your machine:

```shell
git clone https://github.com/${YOUR_GITHUB_USERNAME}/llm-inference.git
cd llm-inference
git remote add upstream https://github.com/OpenCSGs/llm-inference.git
```

_Adding the `upstream` remote sets you up nicely for regularly
[syncing your fork](https://help.github.com/articles/syncing-a-fork/)._

Once you reach this point you are ready to do a full build and deploy as
described below.

## Deploy llm-inference

See [the guide in Readme](https://github.com/OpenCSGs/llm-inference/blob/main/README.md#getting-started) to deploy `llm-inference`.

Note that you need to start `Ray` cluster under `llm-inference` directory.

## Add your model

LLMServe allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, see the [LLMServe Model Registry](./models/README.md).

Then run following command to deploy your model serving.
```
llm-serve start serving --model=<your_model_yaml_path>
```


## Quickly test after deployment

If changes related with CLI `llm-serve`, quickly testing with below:

```
python launch.py ...
```

For example:
```
python launch.py --help

 Usage: launch.py [OPTIONS] COMMAND [ARGS]...

╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --help          Show this message and exit.                                                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ list        List available model(s) and deployed serving etc.                                                                                                      │
│ predict     Predict one or several models with one or multiple prompts, optionally read from file, and save the results to a file.                                 │
│ start       Start application(s) for LLM serving, API server, experimention, fine-tuning and comparation.                                                          │
│ stop        Stop application(s) for LLM serving and API server.                                                                                                    │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

If your changes is related with API Server, need to restart API to active that.
```
llm-serve stop apiserver
llm-serve start apiserver
```

If you would like to use the Restful API to manage your model serving, the API serve is needed.
```
llm-serve start apiserver
```

## Running unit/integration tests

Comming soon.

## Contribute to the code 

Welcome send a PR for your changes, or create a issue for question, problem, or feature requests.

## Feedback 

The best place to provide feedback about the code is via a Github issue. 