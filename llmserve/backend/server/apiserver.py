from ray import serve
import time
import hashlib
from llmserve.backend.logger import get_logger
from fastapi import FastAPI, Body, Header
import threading
from fastapi.middleware.cors import CORSMiddleware
from ray.serve.schema import (
    ServeInstanceDetails,
)
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient

import llmserve.backend.server.config as CONFIG
from llmserve.backend.server.utils import parse_args
from typing import Any, Dict, List, Union, Annotated
from llmserve.common.utils import _replace_prefix, _reverse_prefix
from pydantic import BaseModel, ConfigDict
from llmserve.backend.server.models import (
    LLMApp,
    Scaling_Config_Simple,
)
import ray
from .app import (LLMDeployment, ExperimentalDeployment, RouterDeployment)
from ray.serve._private.constants import (DEFAULT_HTTP_PORT)
from threading import Thread
from ray.serve.deployment import Application
import uuid

class ServeRunThread(Thread):
    def __init__(self, target:Application, name:str, route_prefix:str):
        super().__init__()
        self.target = target
        self.name = name
        self.route_prefix = route_prefix
    
    # overwrite
    def run(self):
        logger.info(f"Server run {self.name} in thread")
        serve.run(target=self.target, name=self.name, route_prefix=self.route_prefix, blocking=False)

#TODO @depengli need reuse the defination in model.py of move to there
class ModelConfig(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )
    model_id: str
    model_task: str
    model_revision: str
    is_oob: bool
    # initialization: InitializationConfig
    scaling_config: Scaling_Config_Simple

#TODO @depengli need reuse the defination in model.py of move to there
class ModelIdentifier(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )
    model_id: str
    model_revision: str = "main"

logger = get_logger("ray.serve")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 5,
        "target_ongoing_requests": 3, # the average number of ongoing requests per replica that the Serve autoscaler tries to ensure
    },
    max_ongoing_requests=5,  # the maximum number of ongoing requests allowed for a replica
)

@serve.ingress(app)
class ApiServer:
    def __init__(self) -> None:
        self.deployments = {}
        self.model_configs = {}
        self.compare_models = []
        self.compare_deployments = {}
        self.compare_model_configs = {}
        # self.newload_model = []
        self.support_models = parse_args("./models")

    def list_deployment_from_ray(self, experimetal: bool) -> List[Any]:
        serve_details = ServeInstanceDetails(**ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
        deployments = []
        if experimetal:
            for key, value in serve_details.applications.items():
                if "apiserver" in key or "cmp_" in key:
                    continue
                apps = value.dict()
                # filtered_deployments= apps.get("deployments").copy()
                filtered_deployments = {}
                deploy_time = apps.get("last_deployed_time_s")
                name = apps.get("name")
                model_id = _replace_prefix(name)
                for k, v in apps.get("deployments").items():
                    if "ExperimentalDeployment" in k:
                        continue
                    v["last_deployed_time_s"] = deploy_time
                    v["id"] = model_id
                    filtered_deployments.update(v.copy())
                deployments.append(filtered_deployments)
        else:
            for key, value in serve_details.applications.items():
                if "cmp_models" not in key:
                    continue
                apps = value.dict()
                deploy_time = apps.get("last_deployed_time_s")
                filtered_deployments = {}
                cmp_models = []
                for k, v in apps.get("deployments").items():
                    if "RouterDeployment" in k:
                        continue
                    model_id = v.get("deployment_config").get(
                        "user_config").get("model_conf").get("model_id")
                    v["last_deployed_time_s"] = deploy_time
                    v["id"] = model_id
                    cmp_models.append(v.copy())

                prefix = apps.get("name").split('_', 2)
                filtered_deployments["url"] = CONFIG.URL + \
                    prefix[0] + "_" + prefix[2]
                filtered_deployments["id"] = prefix[2]
                filtered_deployments["models"] = cmp_models
                deployments.append(filtered_deployments)
        return deployments

    def load_model(self, models: Union[List[str], List[ModelConfig], List[LLMApp]], comparation: bool) -> List[Any]:
        newload_model = []
        self.compare_deployments = {}
        self.compare_model_configs = {}
        mds = parse_args(models)
        if not mds:
            raise RuntimeError("No enabled models were found.")

        for model in mds:
            if model.model_conf.model_id in self.model_configs.keys():
                continue
            name = _reverse_prefix(model.model_conf.model_id)
            user_config = model.dict()
            deployment_config = model.deployment_config.dict()
            deployment_config = deployment_config.copy()
            max_ongoing_requests = deployment_config.pop(
                "max_ongoing_requests", None
            ) or (user_config["model_conf"]["generation"].get("max_batch_size", 1) if user_config["model_conf"]["generation"] else 1)
            deployment = LLMDeployment.options(  # pylint:disable=no-member
                name=name,
                max_ongoing_requests=max_ongoing_requests,
                user_config=user_config,
                **deployment_config,
            ).bind()
            if not comparation:
                self.model_configs[model.model_conf.model_id] = model
                self.deployments[model.model_conf.model_id] = deployment
            else:
                self.compare_model_configs[model.model_conf.model_id] = model
                self.compare_deployments[model.model_conf.model_id] = deployment
            newload_model.append(model.model_conf.model_id)
            logger.info(f"Appended {model.model_conf.model_id}")
        return newload_model

    def load_model_args(self, args: ModelConfig) -> Dict[str, Any]:
        if args.model_id in self.model_configs.keys():
            model = self.model_configs.get(args.model_id)
        else:
            model = CONFIG.EXPERIMENTAL_LLMTEMPLATE
        if args.scaling_config:
            for key, value in args.scaling_config.__dict__.items():
                setattr(model.scaling_config, key, value)
        # if args.initialization.initializer:
        #    for key,value in args.initialization.initializer.__dict__.items():
        #        setattr(model.model_config.initialization.initializer,key,value)
        # if args.initialization.pipeline:
        #    model.model_config.initialization.pipeline =  args.initialization.pipeline
        model.model_conf.model_id = args.model_id
        user_config = model.dict()
        if args.is_oob:
            deployment_config = model.deployment_config.dict()
        else:
            deployment_config = model.deployment_config
        deployment_config = deployment_config.copy()
        max_ongoing_requests = deployment_config.pop(
            "max_ongoing_requests", None
        ) or (user_config["model_conf"]["generation"].get("max_batch_size", 1) if user_config["model_conf"]["generation"] else 1)

        deployment = LLMDeployment.options(  # pylint:disable=no-member
            name=_reverse_prefix(args.model_id),
            max_ongoing_requests=max_ongoing_requests,
            user_config=user_config,
            **deployment_config,
        ).bind()

        serve_name = _reverse_prefix(args.model_id)
        serve_port = user_config["model_conf"]["initialization"]["runtime_env"].get(
            "serve_port", DEFAULT_HTTP_PORT)
        app = ExperimentalDeployment.bind(  # pylint:disable=no-member
            deployment, model)
        ray._private.usage.usage_lib.record_library_usage("llmserve")
        serve.run(app, host=CONFIG.SERVE_RUN_HOST, name=serve_name,
                  route_prefix="/" + serve_name, _blocking=False)
        self.model_configs[args.model_id] = model
        self.deployments[args.model_id] = deployment
        return {"url": "http://" + CONFIG.SERVE_RUN_HOST + ":" + str(serve_port) + "/" + serve_name, "models": self.model_configs}

    @app.post("/start_experimental")
    async def start_experimental(self, models: Union[ModelConfig, str] = Body(..., embed=True)) -> Dict[str, Any]:
        if isinstance(models, ModelConfig) and not models.is_oob:
            return self.load_model_args(models)
        else:
            mods = models
            if isinstance(models, ModelConfig):
                mods = models.model_id
            newload_model = self.load_model(mods, False)
            if newload_model == []:
                return {"response": "No models to load, model already exist."}
            for model in newload_model:
                serve_name = _reverse_prefix(model)
                app = ExperimentalDeployment.bind(self.deployments.get(model), self.model_configs.get(model))
                ray._private.usage.usage_lib.record_library_usage("llmserve")
                serve.run(app, host=CONFIG.SERVE_RUN_HOST, name=serve_name, route_prefix="/" + serve_name, _blocking=False)
        return {"start_experimental": serve_name, "models": self.model_configs}

    @app.post("/start_serving")
    async def start_serving(self, user_name: Annotated[str, Header()], models: Union[List[ModelConfig], ModelConfig] = Body(...)) -> Dict[str, Any]:
        logger.info(f"api start serving {models}")
        models = [models] if isinstance(models, ModelConfig) else models
        self.load_model_for_comparation(models)
        # Create serving one by one under different application
        started_serving = {}
        model_keys = []
        for key, value in self.compare_model_configs.items():
            deployment = {}
            model_config = {}
            deployment[key] = self.compare_deployments[key]
            model_config[key] = value
            app = RouterDeployment.bind(deployment, model_config)
            ray._private.usage.usage_lib.record_library_usage("llmserve")
            # deployment_config = model_config[key].deployment_config.dict()
            user_config = value.dict()
            model_id = _replace_prefix(user_config["model_conf"].get("model_id"))
            # TBD: To add revision to model_config, that's needed to be implemented for CLI (in YAML) and API together.
            # ... For now, that's "main" before implement this.
            model_revision = user_config["model_conf"].get("model_revision", "main")
            model_identifier = model_id.strip() + "-" + model_revision.strip()
            model_hash = hashlib.md5(model_identifier.encode()).hexdigest()[:12]
            serving_name = user_name.strip() + "-" + model_hash
            logger.info(f"Starting serving for {model_identifier} by create thread ...")
            #serve.run(target=app, name=serving_name, route_prefix="/" + serving_name, blocking=False)
            t = ServeRunThread(target=app, name=serving_name, route_prefix="/" + serving_name)
            t.start()
            logger.info(f"Done serving model {model_id} on /{serving_name}")
            started_serving[serving_name] = value
            model_keys.append(key)
        logger.info(f"start all serving {model_keys} done")
        return started_serving

    @app.get("/list_serving")
    async def list_serving(self, user_name: Annotated[str, Header()],
                           models: Union[List[ModelIdentifier],
                                         ModelIdentifier] = Body(default=None)
                           ) -> Dict[str, Any]:
        serving_info = {}
        app_list = []
        if models:
            models = [models] if isinstance(models, ModelIdentifier) else models
            for mod in models:
                model_revision = mod.model_revision if mod.model_revision else "main"
                mod_identifier = mod.model_id.strip() + "-" + model_revision.strip()
                logger.info("Getting serving for {model_identifier} ...")
                model_hash = hashlib.md5(mod_identifier.encode()).hexdigest()[:12]
                app_list.append(user_name.strip() + "-" + model_hash)
        logger.info(f"Begin read ray serve details")
        serve_details = ServeInstanceDetails(**ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
        logger.info(f"End read ray serve details")
        for app_name in serve_details.applications.keys():
            if app_list and app_name not in app_list:
                continue
            if app_name.startswith(user_name + "-"):
                model_info = {}
                model_url = {}
                app_status = ""
                deployment_status = {}
                serving_status = {}
                apps = serve_details.applications[app_name].dict()
                app_status = apps.get("status").value
                route_prefix = apps.get("route_prefix")
                # TBD need get real serve port, need write to ray deployment fristly?
                # serve_port = "8000"
                if "ExperimentalDeployment" in apps.get("deployments").keys():
                    for k, v in apps.get("deployments").items():
                        deployment_status[k] = v.get("status").value
                        if k != "ExperimentalDeployment":
                            model_id = v.get("deployment_config").get("user_config").get("model_conf").get("model_id")
                            model_url[model_id] = "/" + _reverse_prefix(model_id)
                elif "RouterDeployment" in apps.get("deployments").keys():
                    for k, v in apps.get("deployments").items():
                        deployment_status[k] = v.get("status").value
                        if k != "RouterDeployment":
                            model_id = v.get("deployment_config").get("user_config").get("model_conf").get("model_id")
                            model_url[model_id] = route_prefix + "/" + _reverse_prefix(model_id) + "/run/predict"
                else:
                    # Neither ExperimentalDeployment nor RouterDeployment is included in {model}, not a llm-serve application, pass
                    pass
                serving_status[app_name] = {"application_status": app_status, "deployments_status": deployment_status}
                model_info["status"] = serving_status
                model_info["url"] = model_url
                serving_info[app_name] = model_info
        return serving_info

    @app.post("/stop_serving")
    async def stop_serving(self, user_name: Annotated[str, Header()],
                           models: Union[List[ModelIdentifier], ModelIdentifier] = Body(
                               ..., description="Specify model name and revision")
                           ) -> Dict[str, Any]:
        models = [models] if isinstance(models, ModelIdentifier) else models
        stopped_serving = []
        for mod in models:
            model_revision = mod.model_revision if mod.model_revision else "main"
            mod_identifier = mod.model_id.strip() + "-" + model_revision.strip()
            model_hash = hashlib.md5(mod_identifier.encode()).hexdigest()[:12]
            serving_name = user_name.strip() + "-" + model_hash
            logger.info("Stopping serving for {model_identifier} ...")
            serve.delete(serving_name, _blocking=True)
            stopped_serving.append(mod.model_id)
        return {"Stopped Serving": stopped_serving}

    @app.get("/list_deployments")
    async def list_deployments(self) -> List[Any]:
        deployments = self.list_deployment_from_ray(True)
        return deployments

    @app.get("/list_apps")
    async def list_apps(self) -> Dict[str, Any]:
        serve_details = ServeInstanceDetails(**ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
        return serve_details.applications

    @app.get("/oob_models")
    async def list_oob_models(self) -> Dict[str, Any]:
        text, sum, image2text, trans, qa = [], [], [], [], []
        for model in self.support_models:
            if model.model_conf.model_task == "text-generation":
                text.append(model.model_conf.model_id)
            if model.model_conf.model_task == "translation":
                trans.append(model.model_conf.model_id)
            if model.model_conf.model_task == "summarization":
                sum.append(model.model_conf.model_id)
            if model.model_conf.model_task == "question-answering":
                qa.append(model.model_conf.model_id)
            if model.model_conf.model_task == "image-to-text":
                image2text.append(model.model_conf.model_id)
        return {
            "text-generation": text,
            "translation": trans,
            "summarization": sum,
            "question-answering": qa,
            "image-to-text": image2text,
        }

    @app.post("/models")
    async def get_model(self, models: List[str] = Body(..., description="models name")) -> Dict[str, Any]:
        model_config = {}
        for model in models:
            model_config[model] = self.model_configs.get(model)
        return model_config

    @app.post("/update_serving")
    async def update_model(self, model: ModelConfig = Body(..., embed=True)) -> Dict[str, Any]:
        models = self.list_deployment_from_ray(True)
        serve_conf = {"name": _reverse_prefix(model.model_id)}
        for mod in models:
            if model.model_id != mod.get("id"):
                continue
            md = mod.get("deployment_config").get("user_config")
            md = LLMApp(scaling_config=md.get("scaling_config"), model_conf=md.get(
                "model_conf"), deployment_config=md.get("deployment_config"))
            if model.scaling_config:
                for key, value in model.scaling_config.__dict__.items():
                    setattr(md.scaling_config, key, value)

                user_config = md.dict()
                deployment_config = md.deployment_config.dict()  # pylint:disable=no-member
                deployment_config = deployment_config.copy()
                max_ongoing_requests = deployment_config.pop(
                    "max_ongoing_requests", None
                ) or (user_config["model_conf"]["generation"].get("max_batch_size", 1) if user_config["model_conf"]["generation"] else 1)

                deployment = LLMDeployment.options(  # pylint:disable=no-member
                    name=serve_conf["name"],
                    max_ongoing_requests=max_ongoing_requests,
                    user_config=user_config,
                    **deployment_config,
                ).bind()
                app = ExperimentalDeployment.bind(  # pylint:disable=no-member
                    md, deployment)
                ray._private.usage.usage_lib.record_library_usage("llmserve")

                serve.run(app, host=CONFIG.SERVE_RUN_HOST,
                          name=serve_conf["name"], route_prefix="/" + serve_conf["name"], _blocking=False)
                try:
                    serve_port = user_config["model_conf"]["initialization"]["runtime_env"].get(
                        "serve_port", DEFAULT_HTTP_PORT)
                except:
                    serve_port = DEFAULT_HTTP_PORT
        return {"url": "http://" + CONFIG.SERVE_RUN_HOST + ":" + str(serve_port) + "/" + serve_conf["name"], "models": md}

    def load_model_for_comparation(self, models: List[Union[ModelConfig, str]]):
        mods = []
        self.compare_deployments = {}
        self.compare_model_configs = {}

        for model in models:
            logger.info(f"load model: {model}")
            parsed_models = []
            template = []
            if isinstance(model, str):
                logger.info(f"parse model string: {model}")
                parsed_models = parse_args(model)
            else:
                if model.is_oob:
                    logger.info(f"parse oob model_id: {model.model_id}")
                    parsed_models = parse_args(model.model_id)
                else:
                    logger.info(f"parse non-oob model_id: {model.model_id}")
                    template = CONFIG.COMPARATION_LLMTEMPLATE
                    parsed_model = copy.deepcopy(template)
                    parsed_model.model_conf.model_id = model.model_id
                    parsed_models.append(parsed_model)
                # set scaling_config
                if model.scaling_config:
                    for key, value in model.scaling_config.__dict__.items():
                        setattr(parsed_models[0].scaling_config, key, value)
            
            for md in parsed_models:
                user_config = md.dict()
                if model.is_oob:
                    deployment_config = md.deployment_config.dict()
                else:
                    deployment_config = md.deployment_config
                deployment_config = deployment_config.copy()
                max_ongoing_requests = deployment_config.pop(
                    "max_ongoing_requests", None
                ) or (user_config["model_conf"]["generation"].get("max_batch_size", 1) if user_config["model_conf"]["generation"] else 1)
                
                name = _reverse_prefix(md.model_conf.model_id)
                logger.info(f"LLMDeployment.options for {name} with deployment_config={deployment_config}")
                logger.info(f"LLMDeployment.options for {name} with user_config={user_config}")
                deployment = LLMDeployment.options(  # pylint:disable=no-member
                    name=name,
                    max_ongoing_requests=max_ongoing_requests,
                    user_config=user_config,
                    **deployment_config,
                ).bind()

                self.compare_model_configs[md.model_conf.model_id] = md
                self.compare_deployments[md.model_conf.model_id] = deployment
        return

    def run_frontend(self, prefix, compare_prefix):
        logger.info("startting LLMServeFrontend")
        from llmserve.frontend.app import LLMServeFrontend
        ray._private.usage.usage_lib.record_library_usage("llmserve")
        run_duration = 10 * 60
        start_time = time.time()
        while True:
            serve_details = ServeInstanceDetails(
                **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
            app = {}
            for key, value in serve_details.applications.items():
                if compare_prefix not in key:
                    continue
                app = value.dict()
            # logger.info(app)
            if app.get("status") == "RUNNING":
                break
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= run_duration:
                break
            time.sleep(5)
        logger.info(app)

        comparationApp = LLMServeFrontend.options(ray_actor_options={  # pylint:disable=no-member
                                                  "num_cpus": 1}, name="LLMServeFrontend").bind(CONFIG.URL + compare_prefix)
        serve.run(comparationApp, host=CONFIG.SERVE_RUN_HOST,
                  name=prefix, route_prefix="/" + prefix, _blocking=False)

    @app.post("/launch_comparation")
    async def launch_comparation(self, models: List[ModelConfig], user: str = Body(..., embed=True)) -> Dict[str, Any]:
        self.load_model_for_comparation(models)
        app = RouterDeployment.bind(  # pylint:disable=no-member
            self.compare_deployments, self.compare_model_configs)
        logger.info(self.compare_model_configs)
        ray._private.usage.usage_lib.record_library_usage("llmserve")
        prefix = "cmp_models"
        prefix_cmp = "cmp"
        uuid_s = str(uuid.uuid4())

        if user:
            prefix = prefix + "_" + user + "_" + uuid_s[:6]
            prefix_cmp = prefix_cmp + "_" + user + "_" + uuid_s[:6]
        serve.run(app, host=CONFIG.SERVE_RUN_HOST, name=prefix,
                  route_prefix="/" + prefix, _blocking=False)

        thread = threading.Thread(
            target=self.run_frontend, args=(prefix_cmp, prefix))
        thread.daemon = True
        thread.start()
        # await self.run_frontend(prefix_cmp, prefix)
        return {"url": CONFIG.URL + prefix_cmp, "models": self.compare_model_configs, "ids": [prefix, prefix_cmp]}

    @app.post("/update_comparation")
    async def update_comparation(self, models: List[ModelConfig], name: str = Body(..., embed=True)) -> Dict[str, Any]:
        self.load_model_for_comparation(models)
        app = RouterDeployment.bind(  # pylint:disable=no-member
            self.compare_deployments, self.compare_model_configs)
        logger.info(self.compare_model_configs)
        ray._private.usage.usage_lib.record_library_usage("llmserve")
        prefix = "cmp_models"
        prefix_cmp = "cmp"
        if name:
            prefix = prefix + "_" + name
            prefix_cmp = prefix_cmp + "_" + name
        serve.run(app, host=CONFIG.SERVE_RUN_HOST, name=prefix,
                  route_prefix="/" + prefix, _blocking=False)

        thread = threading.Thread(
            target=self.run_frontend, args=(prefix_cmp, prefix))
        thread.daemon = True
        thread.start()
        # await self.run_frontend(prefix_cmp, prefix)
        return {"url": CONFIG.URL + prefix_cmp, "models": self.compare_model_configs, "ids": [prefix, prefix_cmp]}

    @app.get("/models_comparation")
    async def models_comparation(self) -> Dict[str, Any]:
        text = []

        for model in self.support_models:
            if model.model_conf.model_task == "text-generation":
                text.append(model.model_conf.model_id)
        return {
            "text-generation": text,
        }

    @app.get("/list_comparation")
    async def list_comparation(self) -> List[Any]:
        deployments = self.list_deployment_from_ray(False)

        return deployments

    @app.post("/delete_comparation")
    async def delete_app(self, names: List[str] = Body(..., description="model id or all", embed=True)) -> Dict[str, Any]:
        for name in names:
            if "all" in name or "All" in names:
                serve_details = ServeInstanceDetails(
                    **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
                for key, value in serve_details.applications.items():
                    if "cmp_" in key:
                        serve.delete(key, _blocking=False)
            else:
                serve.delete("cmp_models_" + name, _blocking=False)
                serve.delete("cmp_" + name, _blocking=False)
        return {"comparation": "Delete" + name + "Successful"}

    # @app.get("/serving_status")
    # async def serving_status(self, models: List[str] = Body(..., description="models name", embed=True)) -> Dict[str, Any]:
    #    serve_details = ServeInstanceDetails(
    #        **ServeSubmissionClient(CONFIG.RAY_AGENT_ADDRESS).get_serve_details())
    #    serving_status = {}
    #    for model in models:
    #        model_id = _reverse_prefix(model)
    #        app_status = ""
    #        deployment_status = {}
    #        if model_id in serve_details.applications.keys():
    #            apps = serve_details.applications[model_id].dict()
    #            app_status = apps.get("status").value
    #            for k, v in apps.get("deployments").items():
    #                    deployment_status[k] = v.get("status").value
    #            serving_status[model_id] = {"application_status": app_status, "deployments_status": deployment_status}
    #    return serving_status
    
    # @app.get("/metadata")
    # async def metadata(self, models: List[str] = Body(..., description="models name", embed=True)) -> Dict[str, Any]:
    #    metadata = {}
    #    for model in models:
    #        #model = _replace_prefix(model)
    #        metadata = self.model_configs[model].dict(
    #            exclude={
    #                "model_config": {"initialization": {"s3_mirror_config", "runtime_env"}}
    #            }
    #        )
    #        logger.info(metadata)
    #        metadata[model] = metadata
    #    return metadata