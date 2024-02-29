import sys
from typing import List, Union, Optional, Dict

import ray._private.usage.usage_lib
from ray import serve

from llmserve.backend.server.app import LLMDeployment, RouterDeployment, ExperimentalDeployment, ApiServer
# from llmserve.backend.server.app import ApplicationDeployment
from llmserve.backend.server.models import LLMApp, ServeArgs, FTApp
from llmserve.backend.server.utils import parse_args, parse_args_ft, get_serve_port
import uuid
import os
from llmserve.backend.llm.ft import TransformersFT
from llmserve.backend.logger import get_logger
from ray.serve._private.constants import ( DEFAULT_HTTP_PORT )
from llmserve.common.utils import _replace_prefix, _reverse_prefix

# ray.init(address="auto")
logger = get_logger(__name__)

SERVE_RUN_HOST = "0.0.0.0"

def get_serve_start_port(port: int): 
    serve_start_port = port
    serve_runtime_port = get_serve_port()
    if serve_runtime_port > -1:
        logger.info(f"Serve is already running at {SERVE_RUN_HOST}:{serve_runtime_port}")
        serve_start_port = serve_runtime_port
    return serve_start_port

def llm_server(args: Union[str, LLMApp, List[Union[LLMApp, str]]]):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    models = parse_args(args)
    if not models:
        raise RuntimeError("No enabled models were found.")

    # For each model, create a deployment
    deployments = {}
    model_configs = {}
    for model in models:
        if model.model_config.model_id in model_configs:
            raise ValueError(
                f"Duplicate model_id {model.model_config.model_id} specified. "
                "Please ensure that each model has a unique model_id. "
                "If you want two models to share the same Hugging Face Hub ID, "
                "specify initialization.hf_model_id in the model config."
            )
        logger.info(f"Initializing LLM app {model.json(indent=2)}")
        user_config = model.dict()
        deployment_config = model.deployment_config.dict()
        model_configs[model.model_config.model_id] = model
        deployment_config = deployment_config.copy()
        max_concurrent_queries = deployment_config.pop(
            "max_concurrent_queries", None
        ) or user_config.get("model_config", {}).get("generation", {}).get("max_batch_size", 1)

        deployments[model.model_config.model_id] = LLMDeployment.options(
            name=_reverse_prefix(model.model_config.model_id),
            max_concurrent_queries=max_concurrent_queries,
            user_config=user_config,
            **deployment_config,
        ).bind()
    # test = []
    return RouterDeployment.bind(deployments, model_configs)

def llm_experimental(args: Union[str, LLMApp, List[Union[LLMApp, str]]]):
    """Serve LLM Models

    This function returns a Ray Serve Application.

    Accepted inputs:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    You can use `serve.run` to run this application on the local Ray Cluster.

    `serve.run(llm_backend(args))`.

    You can also remove
    """
    models = parse_args(args)
    if not models:
        raise RuntimeError("No enabled models were found.")

    # For model, create a deployment
    model = models[0]

    if isinstance(model, LLMApp):
        logger.info(f"Initialized a LLM app instance of LLMApp {model.json(indent=2)}")
    else:
        raise RuntimeError("Not a LLM app instance were found.")
    
    user_config = model.dict()
    deployment_config = model.deployment_config.dict()
    deployment_config = deployment_config.copy()
    max_concurrent_queries = deployment_config.pop(
        "max_concurrent_queries", None
    ) or (user_config["model_config"]["generation"].get("max_batch_size", 1) if user_config["model_config"]["generation"] else 1)
    
    deployment = LLMDeployment.options(
        name=_reverse_prefix(model.model_config.model_id),
        max_concurrent_queries=max_concurrent_queries,
        user_config=user_config,
        **deployment_config,
    ).bind()
    serve_conf = {
        "name": _reverse_prefix(model.model_config.model_id)
    }

    return (ExperimentalDeployment.bind(deployment, model), serve_conf)

def llm_application(args):
    """This is a simple wrapper for LLM Server
    That is compatible with the yaml config file format

    """
    serve_args = ServeArgs.parse_obj(args)
    return llm_server(serve_args.models)[0]

def run(models: Union[LLMApp, str], appname: str = None, port: int= DEFAULT_HTTP_PORT):
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    app = llm_server(list(models))
    ray._private.usage.usage_lib.record_library_usage("llmserve")
    serve_start_port = get_serve_start_port(port)
    serve.start(http_options={"host": SERVE_RUN_HOST, "port": serve_start_port})
    if not appname:
        appname = "default"
    serve.run(app, name = appname, route_prefix = "/api/v1/" + appname)


def run_experimental(models: Union[LLMApp, str], appname: str = None, port: int = DEFAULT_HTTP_PORT):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    app = llm_experimental(list(models))
    serve_conf = app[1]
    ray._private.usage.usage_lib.record_library_usage("llmserve")
    serve_start_port = get_serve_start_port(port)
    logger.info(f"Serve run app at {SERVE_RUN_HOST}:{serve_start_port}/{serve_conf['name']}")
    serve.start(http_options={"host": SERVE_RUN_HOST, "port": serve_start_port})
    serve_name = appname + "-" + serve_conf["name"] if appname else serve_conf["name"]
    serve.run(app[0], name=serve_name, route_prefix="/" + serve_name)

def del_serve(app_name: str):
    serve.delete(app_name, _blocking = True)

def start_apiserver(port: int = DEFAULT_HTTP_PORT):
    """Run the API Server on the local Ray Cluster

    Args:
        *host: The host ip to run.
        *port: The port to run.     
    
    """
    app = ApiServer.bind()
   
    ray._private.usage.usage_lib.record_library_usage("llmserve")
    #ray.init(address="auto")
    serve_start_port = get_serve_start_port(port)
    serve.start(http_options={"host": SERVE_RUN_HOST, "port": serve_start_port})
    logger.info(f"Serve 'apiserver' is running at {SERVE_RUN_HOST}/{serve_start_port}")
    serve.run(app, name="apiserver", route_prefix="/api")

def run_ft(ft: Union[FTApp, str]):
    """Run the LLM Server on the local Ray Cluster

    Args:
        model: A LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/model.yaml") # run one model in the model directory
       run(FTApp)         # run a single LLMApp
    """

    ft = parse_args_ft(ft)
    if not ft:
        raise RuntimeError("No valiabled fine tune defination were found.")
    
    if isinstance(ft, FTApp):
        logger.info(f"Initialized a Finetune instance of FTApp {ft.json(indent=2)}")
    else:
        raise RuntimeError("Not a Finetune App were found.")
    
    ray._private.usage.usage_lib.record_library_usage("llmserve")

    runner = TransformersFT(ft)
    runner.train()
    
def run_comparation():
    """Run the LLM Server on the local Ray Cluster

    Args:
        *models: A list of LLMApp objects or paths to yaml files defining LLMApps

    Example:
       run("models/")           # run all models in the models directory
       run("models/model.yaml") # run one model in the model directory
       run({...LLMApp})         # run a single LLMApp
       run("models/model1.yaml", "models/model2.yaml", {...LLMApp}) # mix and match
    """
    from llmserve.frontend.app import LLMServeFrontend
    serve_runtime_port = get_serve_port()
    cmp_serve_port = DEFAULT_HTTP_PORT
    if serve_runtime_port > -1:
        logger.info(f"Serve is already running at {SERVE_RUN_HOST}:{serve_runtime_port}")
        cmp_serve_port = serve_runtime_port
        serve.start(http_options={"host": SERVE_RUN_HOST, "port": serve_runtime_port})
    else:
        serve.start(http_options={"host": SERVE_RUN_HOST, "port": DEFAULT_HTTP_PORT})
    cmp_address = "/api/v1/default"
    logger.info(f"Bind LLMServeFrontend at {cmp_address}")
    app = LLMServeFrontend.options(ray_actor_options={"num_cpus": 1}, name="LLMServeFrontend").bind(cmp_address)
    ray._private.usage.usage_lib.record_library_usage("llmserve")
    logger.info(f"Serve run comparation running at {SERVE_RUN_HOST}/{cmp_serve_port}")
    serve.run(app, name = "cmp_default", route_prefix="/cmp_default")

if __name__ == "__main__":
    run(*sys.argv[1:])


