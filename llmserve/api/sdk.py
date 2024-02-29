from typing import Annotated, List, Any, Dict, Optional, Union
import glob
import yaml

from ray.serve._private.constants import ( DEFAULT_HTTP_PORT )
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient
from ray.serve.schema import ServeInstanceDetails
from llmserve.common.utils import _replace_prefix, _reverse_prefix

# Preformance issue if import llmserve.backend.server.config
# The address may be defined by user when start ray, better if user can pass this.
#import llmserve.backend.server.config as CONFIG
RAY_AGENT_ADDRESS = "http://localhost:52365"
SERVE_RUN_HOST = "0.0.0.0"

try:
    from langchain.llms import OpenAIChat

    LANGCHAIN_INSTALLED = True
    LANGCHAIN_SUPPORTED_PROVIDERS = {"openai": OpenAIChat}
except ImportError:
    LANGCHAIN_INSTALLED = False

from llmserve.api.env import assert_has_backend

__all__ = ["models", "metadata", "query", "batch_query", "run"]


def list_models(path: str = "./models") -> Dict[str, Dict[str, Any]]:
    files = glob.glob(path + "/*.yaml")
    models = {}
    #metadata = {}
    for file in files:
        with open(file, 'r') as stream:
            try:
                model_config = yaml.safe_load(stream).get('model_config')
                model_id = model_config.get('model_id')
                models[model_id] = model_config
                #print(model_id)
            except yaml.YAMLError as exc:
                print(exc)
    return models


def models() -> List[str]:
    """List available models"""
    from llmserve.common.backend import get_llmserve_backend

    backend = get_llmserve_backend()
    return backend.models()


def _is_llmserve_model(model: str) -> bool:
    """
    Determine if this is an llmserve model. LLMServe
    models do not have a '://' in them.
    """
    return "://" not in model


def _supports_batching(model: str) -> bool:
    provider, _ = model.split("://", 1)
    return provider != "openai"


def _get_langchain_model(model: str):
    if not LANGCHAIN_INSTALLED:
        raise ValueError(
            f"Unsupported model {model}. If you want to use a langchain-"
            "compatible model, install langchain ( pip install langchain )."
        )

    provider, model_name = model.split("://", 1)
    if provider not in LANGCHAIN_SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown model provider for {model}. Supported providers are: "
            f"{' '.join(LANGCHAIN_SUPPORTED_PROVIDERS.keys())}"
        )
    return LANGCHAIN_SUPPORTED_PROVIDERS[provider](model_name=model_name)


def _convert_to_llmserve_format(model: str, llm_result):
    generation = llm_result.generations
    result_list = [{"generated_text": x.text} for x in generation[0]]
    return result_list


def metadata(model_id: str) -> Dict[str, Dict[str, Any]]:
    """Get model metadata"""
    from llmserve.common.backend import get_llmserve_backend

    backend = get_llmserve_backend()
    return backend.metadata(model_id)


def query(model: str, prompt: str, port: int = DEFAULT_HTTP_PORT, appname: Optional[str] = None) -> Dict[str, Union[str, float, int]]:
    """Query LLMServe"""
    from llmserve.common.backend import get_llmserve_backend

    if _is_llmserve_model(model):
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient(RAY_AGENT_ADDRESS).get_serve_details())
        if not appname:
            appname = "default"
        if appname in serve_details.applications.keys():
            if _reverse_prefix(model) in serve_details.applications[appname].dict().get("deployments").keys():
                route_prefix = serve_details.applications[appname].dict().get("route_prefix")
            else:
                raise Exception(f"No {model} deployed under application {appname}. Ensure specify reasonable application and model name.")
        else:
            raise Exception(f"Specify reasonable application name for the model {model}.")
        backend = get_llmserve_backend(port = port,url=route_prefix)
        return backend.completions(prompt, model)
    llm = _get_langchain_model(model)
    return llm.predict(prompt)

def batch_query(
    model: str, prompts: List[str], port: int = DEFAULT_HTTP_PORT, appname: Optional[str] = None,
) -> List[Dict[str, Union[str, float, int]]]:
    """Batch Query LLMServe"""
    from llmserve.common.backend import get_llmserve_backend

    if _is_llmserve_model(model):
        serve_details = ServeInstanceDetails(
            **ServeSubmissionClient(RAY_AGENT_ADDRESS).get_serve_details())
        if not appname:
            appname = "default"
        if appname in serve_details.applications.keys():
            if _reverse_prefix(model) in serve_details.applications[appname].dict().get("deployments").keys():
                route_prefix = serve_details.applications[appname].dict().get("route_prefix")
            else:
                raise Exception(f"No {model} deployed under application {appname}. Ensure specify reasonable application and model name.")
        else:
            raise Exception(f"Specify reasonable application name for the model {model}.")
        backend = get_llmserve_backend(port = port,url = route_prefix)
        return backend.batch_completions(prompts, model)
    else:
        llm = _get_langchain_model(model)
        if _supports_batching(model):
            result = llm.generate(prompts)
            converted = _convert_to_llmserve_format(model, result)
        else:
            result = [{"generated_text": llm.predict(prompt)} for prompt in prompts]
            converted = result
        return converted


def run(model: str, appname: str = None, port: int = DEFAULT_HTTP_PORT) -> None:
    """Run LLMServe on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import run

    run(models=model, appname=appname, port=port)

def run_experimental(model: str, appname: str = None, port: int = DEFAULT_HTTP_PORT) -> None:
    """Run LLMServe on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import run_experimental

    run_experimental(models = model, appname = appname, port = port)

def start_apiserver(port: int = DEFAULT_HTTP_PORT) -> None:
    """Run Api server on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import start_apiserver

    start_apiserver(port)

def del_serve(app_name: str) -> None:
    """Delete ray serve on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import del_serve

    del_serve(app_name)
    
def run_application(flow: dict) -> None:
    """Run LLMServe on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import run_application

    run_application(flow)

def run_ft(ft: str) -> None:
    """Run LLMServe on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import run_ft

    run_ft(ft)

def run_comparation() -> None:
    """Run LLMServe on the local ray cluster

    NOTE: This only works if you are running this command
    on the Ray or Anyscale cluster directly. It does not
    work from a general machine which only has the url and token
    for a model.
    """
    assert_has_backend()
    from llmserve.backend.server.run import run_comparation

    run_comparation()

    
def list_serving(appname: Optional[List[str]]) -> Dict[str, Any]:
    '''Get the serving status and URL for deployed model.'''
    serving_info = {}
    serve_details = ServeInstanceDetails(
        **ServeSubmissionClient(RAY_AGENT_ADDRESS).get_serve_details())
    if not appname:
        all_apps = list(serve_details.applications.keys())
        if "apiserver" in all_apps:
            all_apps.remove("apiserver")
        appname = all_apps
    for app in appname:
        model_info = {}
        model_url = {}
        app_status = ""
        deployment_status = {}
        serving_status = {}
        if app in serve_details.applications.keys():
            apps = serve_details.applications[app].dict()

            app_status = apps.get("status").value
            #for k, v in apps.get("deployments").items():
            #        deployment_status[k] = v.get("status").value

            route_prefix = apps.get("route_prefix")
            #TBD need get real serve port, need write to ray deployment fristly?
            serve_port = "8000"
            if "ExperimentalDeployment" in apps.get("deployments").keys():
                for k, v in apps.get("deployments").items():
                    deployment_status[k] = v.get("status").value
                    if k != "ExperimentalDeployment":
                        model_id = v.get("deployment_config").get("user_config").get("model_config").get("model_id")
                        model_url[model_id] = "http://" + SERVE_RUN_HOST + ":" + serve_port + "/" + _reverse_prefix(model_id)
            elif "RouterDeployment" in apps.get("deployments").keys():
                for k, v in apps.get("deployments").items():
                    deployment_status[k] = v.get("status").value
                    if k != "RouterDeployment":
                        model_id = v.get("deployment_config").get("user_config").get("model_config").get("model_id")
                        model_url[model_id] = "http://" + SERVE_RUN_HOST + ":" + serve_port + route_prefix + "/" + _reverse_prefix(model_id) + "/run/predict"
            else:
                # Neither ExperimentalDeployment nor RouterDeployment is included in {model}, not a llm-serve application, pass
                pass
            serving_status[app] = {"application_status": app_status, "deployments_status": deployment_status}
            model_info["status"] = serving_status
            model_info["url"] = model_url
            serving_info[app] = model_info
    return serving_info
