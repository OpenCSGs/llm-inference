
from llmserve.backend.server.models import LLMApp
import os


LLMTEMPLATE_DEPLOYMENT_CONFIG = {
    "autoscaling_config":{
        "min_replicas": 0,
        "initial_replicas": 1,
        "max_replicas": 8,
        "target_num_ongoing_requests_per_replica": 1.0,
        "metrics_interval_s": 10.0,
        "look_back_period_s": 30.0,
        "smoothing_factor": 1.0,
        "downscale_delay_s": 300.0,
        "upscale_delay_s": 90.0,
    },
    "ray_actor_options": {
        "num_cpus": 0.1  
    }
}
LLMTEMPLATE_MODEL_CONFIG_COMPARATION = {
    "warmup": True,
    "model_task": "text-generation",
    "model_id": "template",
    "max_input_words": 800,
    "initialization": {
        "runtime_env": {
        },
        "initializer":{
            "type": "SingleDevice",
            "dtype": "float32",
            "from_pretrained_kwargs":{
                "use_cache": True ,
                "trust_remote_code": True
            }
            
        },
        "pipeline": "default"
    },
    "generation":{
        "max_batch_size": 18,
        "generate_kwargs":{
            "do_sample": True,
            "max_new_tokens": 128,
            "min_new_tokens": 16,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "top_p": 0.8,
            "top_k": 50,
        },             
        "prompt_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n",
        "stopping_sequences": ["### Response:", "### End"]
    }                            
}

# TODO defaulttransformers leverage transformer pipeline to load the model, it's a problem, since some model cannot load by pipeline
LLMTEMPLATE_MODEL_CONFIG_EXPERIMENTAL = {
    "warmup": True,
    "model_task": "text-generation",
    "model_id": "template",
    "max_input_words": 800,
    "initialization": {
        "runtime_env": {
        },
        "initializer":{
            "type": "TransformersPipeline",
            "dtype": "float32",
            "use_fast": False,
            "from_pretrained_kwargs":{
                "use_cache": True ,
                "trust_remote_code": True
            }
            
        },
        "pipeline": "defaulttransformers"
    },
    "generation":{
        "max_batch_size": 18,
        "generate_kwargs":{
            "do_sample": True,
            "max_new_tokens": 128,
            "min_new_tokens": 16,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "top_p": 0.8,
            "top_k": 50,
        },             
        "prompt_format": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n### Instruction:\n{instruction}\n### Response:\n",
        "stopping_sequences": ["### Response:", "### End"]
    }                            
}


LLMTEMPLATE_SCALE_CONFIG = {
    "num_workers":1, 
    "num_gpus_per_worker":0.0, 
    "num_cpus_per_worker":1.0, 
    "placement_strategy":'PACK', 
    "resources_per_worker":None, 
    "pg_timeout_s":600
}
EXPERIMENTAL_LLMTEMPLATE = LLMApp(scaling_config=LLMTEMPLATE_SCALE_CONFIG.copy(),model_config=LLMTEMPLATE_MODEL_CONFIG_EXPERIMENTAL.copy())
EXPERIMENTAL_LLMTEMPLATE.deployment_config = LLMTEMPLATE_DEPLOYMENT_CONFIG.copy()

COMPARATION_LLMTEMPLATE = LLMApp(scaling_config=LLMTEMPLATE_SCALE_CONFIG.copy(),model_config=LLMTEMPLATE_MODEL_CONFIG_COMPARATION.copy())
COMPARATION_LLMTEMPLATE.deployment_config = LLMTEMPLATE_DEPLOYMENT_CONFIG.copy()

RAY_AGENT_ADDRESS = "http://localhost:52365"

MODELS_MAPPING = {
    "gpt2": "./models/text-generation--gpt2.yaml",
    "AIWizards/gpt2": "./models/text-generation--gpt2.yaml",
    "t5-small": "./models/translation--t5-small.yaml",
    "THUDM/chatglm2-6b": "./models/text-generation--THUDM-chatglm2-6b.yaml",
    "THUDM/chatglm3-6b": "./models/text-generation--THUDM-chatglm3-6b.yaml",
    "Qwen/Qwen-7B": "./models/text-generation--Qwen--Qwen-7B.yaml",
    "Qwen/Qwen-7B-Chat": "./models/text-generation--Qwen--Qwen-7B-Chat.yaml",
    "LinkSoul/Chinese-Llama-2-7b": "./models/text-generation--LinkSoul--Chinese-Llama-2-7b.yaml",
    "bigscience/bloom-560m": "./models/text-generation--bigscience--bloom-560m.yaml",
    "baichuan-inc/Baichuan-7B": "./models/text-generation--baichuan-inc--Baichuan-7B.yaml",
    "distilbert-base-uncased-finetuned-sst-2-english": "./models/text-classification--distilbert-base-uncased-finetuned-sst-2-english.yaml",
    "facebook/bart-large-cnn": "./models/summarization--facebook--bart-large-cnn.yaml",
    "deepset/roberta-base-squad2": "./models/question-answering--deepset--roberta-base-squad2.yaml",
    "nlpconnect/vit-gpt2-image-captioning": "./models/image-to-text--nlpconnect--vit-gpt2-image-captioning.yaml",
    "facebook/opt-125m": "./models/text-generation--facebook--opt-125m.yaml",
    "facebook/opt-125m-pipeline": "./models/text-generation--facebook--opt-125m-pipeline.yaml",
    "opencsg/opencsg-CodeLlama-7b-v0.1": "./models/text-generation--opencsg--opencsg-CodeLlama-7b-v0.1-pipeline.yaml",
    "OpenCSG/opencsg-CodeLlama-7b-v0.1": "./models/text-generation--opencsg--opencsg-CodeLlama-7b-v0.1-pipeline.yaml",
    "opencsg/opencsg-starcoder-v0.1": "./models/text-generation--opencsg--opencsg-starcoder-15B-v0.1-pipeline.yaml",
    "OpenCSG/opencsg-starcoder-v0.1": "./models/text-generation--opencsg--opencsg-starcoder-15B-v0.1-pipeline.yaml"
}

SERVE_RUN_HOST = "0.0.0.0"
URL = "http://127.0.0.1:8000/"

