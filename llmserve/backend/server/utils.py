import os
from typing import List, Union, Dict, Any

from llmserve.backend.logger import get_logger
import llmserve.backend.server.config as CONFIG
from llmserve.backend.server.models import LLMApp, FTApp
import gradio as gr
from ray.serve.context import _get_global_client
from ray.serve.exceptions import RayServeException
from ray.serve._private.constants import ( SERVE_NAMESPACE )

logger = get_logger(__name__)

def parse_args(args: Union[str, LLMApp, List[Union[LLMApp, str]]]) -> List[LLMApp]:
    """Parse the input args and return a standardized list of LLMApp objects

    Supported args format:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    """
    raw_models = []
    if isinstance(args, list):
        raw_models = args
    else:
        raw_models = [args]
    
    logger.info(f"Parsing model args {raw_models}")
    # For each
    models: List[LLMApp] = []
    for raw_model in raw_models:
        if isinstance(raw_model, str):
            parsed_models = _parse_path_args(raw_model, False)
        else:
            parsed_models = [LLMApp.parse_obj(raw_model)]
        models += parsed_models
    return [model for model in models if model.enabled]


def _parse_path_args(path: str, isFt: bool) -> List[LLMApp]:
    logger.info(f"Parsing model path {path}")
    if "models/" in path:
        assert os.path.exists(
            path
        ), f"Could not load model from {path}, as it does not exist."

    if os.path.isfile(path):
        with open(path, "r") as f:
            return [LLMApp.parse_yaml(f)]
    elif os.path.isdir(path):
        apps = []
        for root, _dirs, files in os.walk(path):
            for p in files:
                if not isFt:
                    if p.startswith("ft-" ):
                        continue 
                if _is_yaml_file(p):
                    with open(os.path.join(root, p), "r") as f:
                        logger.info(f"Found model file {f.name}")
                        apps.append(LLMApp.parse_yaml(f))                 
        return apps
    else:
        file_name = CONFIG.MODELS_MAPPING.get(path)
        if file_name:
            with open(file_name, "r") as f:
                return [LLMApp.parse_yaml(f)]
        else:
            raise ValueError(
                f"Could not load model from directory <./models/>"
            )

def parse_args_ft(args: Union[str, FTApp]) -> FTApp:
    """Parse the input args and return a standardized list of LLMApp objects

    Supported args format:
    1. The path to a yaml file defining your LLMApp
    2. The path to a folder containing yaml files, which define your LLMApps
    2. A list of yaml files defining multiple LLMApps
    3. A dict or LLMApp object
    4. A list of dicts or LLMApp objects

    """
    if isinstance(args, str):
        parsed_ft = _parse_path_args_ft(args)
    else:
        parsed_ft = FTApp.parse_obj(args)

    return parsed_ft


def _parse_path_args_ft(path: str) -> FTApp:
    assert os.path.exists(
        path
    ), f"Could not load model from {path}, as it does not exist."
    if os.path.isfile(path):
        with open(path, "r") as f:
            return FTApp.parse_yaml(f)
    else:
        raise ValueError(
            f"Could not load model from {path}, as it is not a file or directory."
        )


def _is_yaml_file(filename: str) -> bool:
    yaml_exts = [".yml", ".yaml", ".json"]
    for s in yaml_exts:
        if filename.endswith(s):
            return True
    return False

def render_gradio_params(hg_task: str) -> Dict[str, Any]:
    # customize gradio by hg task
    if hg_task == "audio-classification":
        pipeline_info = {
            "inputs": gr.components.Audio(
                source="microphone", type="numpy", label="Input"
            ),
            "outputs": gr.components.Label(label="Class"),
            "preprocess": lambda i: {"inputs": i},
            "postprocess": lambda r: {i["label"].split(", ")[0]: i["score"] for i in r},
        }
    elif hg_task == "automatic-speech-recognition":
        pipeline_info = {
            "inputs": gr.components.Audio(
                source="microphone", type="numpy", label="Input"
            ),
            "outputs": gr.components.Textbox(label="Output"),
            "preprocess": lambda i: {"inputs": i},
            "postprocess": lambda r: r["text"],
        }
    elif hg_task == "feature-extraction":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Dataframe(label="Output"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r[0],
            "warmup": "Hello, my dog is cute",
        }
    elif hg_task == "fill-mask":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Label(label="Classification"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: {i["token_str"]: i["score"] for i in r},
        }
    elif hg_task == "image-classification":
        pipeline_info = {
            "inputs": gr.components.Image(type="pil", label="Input Image"),
            "outputs": gr.components.Label(type="confidences", label="Classification"),
            "preprocess": lambda i: {"images": i},
            "postprocess": lambda r: {i["label"].split(", ")[0]: i["score"] for i in r},
        }
    elif hg_task == "question-answering":
        pipeline_info = {
            "inputs": [
                gr.components.Textbox(lines=7, label="Context"),
                gr.components.Textbox(label="Question"),
            ],
            "outputs": [
                gr.components.Textbox(label="Answer"),
                gr.components.Label(label="Score"),
            ],
            "preprocess": lambda c, q: {"context": c, "question": q},
            "postprocess": lambda r: (r["answer"], r["score"]),
        }
    elif hg_task == "summarization":
        pipeline_info = {
            "inputs": gr.components.Textbox(lines=7, label="Input"),
            "outputs": gr.components.Textbox(label="Summary"),
            "preprocess": lambda x: {"inputs": x},
            "postprocess": lambda r: r[0]["summary_text"],
            "warmup": """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
                        A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
                        Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
                        In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
                        Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
                        2010 marriage license application, according to court documents.
                        Prosecutors said the marriages were part of an immigration scam.
                        On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
                        After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
                        Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
                        All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
                        Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
                        Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
                        The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
                        Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
                        Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
                        If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
                        """,
        }
    elif hg_task == "text-classification":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Label(label="Classification"),
            "preprocess": lambda x: [x],
            "postprocess": lambda r: {i["label"].split(", ")[0]: i["score"] for i in r},
            "warmup": "Hello, my dog is cute",
            
        }
    elif hg_task == "text-generation":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Textbox(label="Output"),
            "preprocess": lambda x: {"text_inputs": [(text + "\n") for text in x]},
            # "postprocess": lambda r: r[0]["generated_text"],
            "postprocess": lambda r: [text[0]['generated_text'] for text in r],
            "warmup": "Write a short story."
        }
    elif hg_task == "translation":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Textbox(label="Translation"),
            "preprocess": lambda x: [x],
            "postprocess": lambda r: r[0]["translation_text"],
            "warmup": "My name is Wolfgang and I live in Berlin",
        }
    elif hg_task == "text2text-generation":
        pipeline_info = {
            "inputs": gr.components.Textbox(label="Input"),
            "outputs": gr.components.Textbox(label="Generated Text"),
            "preprocess": lambda x: [x],
            "postprocess": lambda r: r[0]["generated_text"],
        }
    elif hg_task == "zero-shot-classification":
        pipeline_info = {
            "inputs": [
                gr.components.Textbox(label="Input"),
                gr.components.Textbox(label="Possible class names (" "comma-separated)"),
                gr.components.Checkbox(label="Allow multiple true classes"),
            ],
            "outputs": gr.components.Label(label="Classification"),
            "preprocess": lambda i, c, m: {
                "sequences": i,
                "candidate_labels": c,
                "multi_label": m,
            },
            "postprocess": lambda r: {
                r["labels"][i]: r["scores"][i] for i in range(len(r["labels"]))
            },
        }
    elif hg_task == "document-question-answering":
        pipeline_info = {
            "inputs": [
                gr.components.Image(type="pil", label="Input Document"),
                gr.components.Textbox(label="Question"),
            ],
            "outputs": gr.components.Label(label="Label"),
            "preprocess": lambda img, q: {"image": img, "question": q},
            "postprocess": lambda r: {i["answer"]: i["score"] for i in r},
        }
    elif hg_task == "visual-question-answering":
        pipeline_info = {
            "inputs": [
                gr.components.Image(type="pil", label="Input Image"),
                gr.components.Textbox(label="Question"),
            ],
            "outputs": gr.components.Label(label="Score"),
            "preprocess": lambda img, q: {"image": img, "question": q},
            "postprocess": lambda r: {i["answer"]: i["score"] for i in r},
        }
    elif hg_task == "image-to-text":
        pipeline_info = {
            "inputs": gr.components.Image(type="pil", label="Input Image"),
            "outputs": gr.components.Textbox(label="Text"),
            "preprocess": lambda i: {"images": [img for img in i]},
            # "postprocess": lambda r: r[0]["generated_text"],
            "postprocess": lambda r: [text[0]['generated_text'] for text in r],
        }
    else:
        raise ValueError(f"Unsupported task type: {hg_task}")
    
    return pipeline_info

def get_serve_port() -> int:
    try:
        client = _get_global_client(_health_check_controller=True)
        logger.info(f"Found existing Serve app in namespace {SERVE_NAMESPACE}")
        serve_detail: Dict = client.get_serve_details()
        if serve_detail and serve_detail["http_options"]:
            return int(serve_detail["http_options"]["port"])
    except RayServeException:
        pass
    return -1