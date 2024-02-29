import asyncio
import os
import subprocess
import time
import traceback
from collections import defaultdict
from functools import wraps
from typing import List, Optional

from transformers import AutoConfig
import torch.distributed as dist
from filelock import FileLock
from ray.air.util.torch_dist import (
    ActorHandle,
    _get_node_and_gpu_ids,
    _init_torch_distributed,
    get_address_and_port,
)
from torch.hub import _get_torch_home

from llmserve.backend.logger import get_logger
from llmserve.backend.server.models import S3MirrorConfig

logger = get_logger(__name__)


def download_model(
    model_id: str,
    endpoint_url: str,
    bucket_uri: str,
    git_uri: str,
    s3_sync_args: Optional[List[str]] = None,
) -> None:
    """
    Download a model from an S3 bucket and save it in TRANSFORMERS_CACHE for
    seamless interoperability with Hugging Face's Transformers library.

    The downloaded model must have a 'hash' file containing the commit hash corresponding
    to the commit on Hugging Face Hub.
    """
    from transformers.utils.hub import TRANSFORMERS_CACHE
    
    isAutoLoadConfigSuccess = False
    modelConfig = None
    try:
        modelConfig = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        isAutoLoadConfigSuccess = True
    except Exception:
        isAutoLoadConfigSuccess = False
    
    if modelConfig and isAutoLoadConfigSuccess:
        logger.info(f"Model exist and success to load AutoConfig from_pretrained '{model_id}'")
        return
    else:
        logger.info(f"Fail to load AutoConfig from_pretrained '{model_id}'")
    
    isS3 = False
    isLocal = False
    isGit = False
    git_temp_dir = "/tmp"
    
    if bucket_uri:
        if bucket_uri.startswith('s3://'):
            isS3 = True
        else:
            isLocal = True
        logger.info(f"Downloading '{model_id}' from '{bucket_uri}' to '{TRANSFORMERS_CACHE}'")
    elif git_uri:
        isGit = True
        logger.info(f"Downloading '{model_id}' from '{git_uri}' to '{TRANSFORMERS_CACHE}'")
    
    path = os.path.expanduser(os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}"))
    s3_sync_args = s3_sync_args or []
    if isS3:
        model_hash_file = os.path.join(bucket_uri, "hash")
        if endpoint_url:
            logger.info(f"Downloading '{model_id}' hash from server '{endpoint_url}' '{model_hash_file}' ")
            subprocess.run(["aws", "--endpoint-url", endpoint_url, "s3", "cp", "--quiet"] + s3_sync_args + [model_hash_file, "."])
        else:
            logger.info(f"Downloading '{model_id}' hash from '{model_hash_file}' ")
            subprocess.run(["aws", "s3", "cp", "--quiet"] + s3_sync_args + [model_hash_file, "."])
    elif isLocal:
        model_hash_file = os.path.join(bucket_uri, "hash")
        logger.info(f"Downloading '{model_id}' hash from '{model_hash_file}' ")
        subprocess.run(["cp -rf " + model_hash_file + " ."], shell=True)
    elif isGit:
        model_tmp_dir = os.path.join(git_temp_dir, model_id)
        logger.info(f"Downloading '{model_id}' from '{git_uri}' to '{model_tmp_dir}'")
        subprocess.run(["git clone " + git_uri + " " + model_tmp_dir], shell=True)
        subprocess.run(["git --git-dir=" + model_tmp_dir + "/.git log  -1 --pretty=format:%H > ./hash"], shell=True)
    
    if not os.path.exists(os.path.join(".", "hash")):
        raise RuntimeError("Hash file not found in the bucket or bucket could not have been downloaded.")
    
    with open(os.path.join(".", "hash"), "r") as f:
        f_hash = f.read().strip()
    
    model_cache_path = os.path.join(path, "snapshots", f_hash)
    
    model_config_file = os.path.join(model_cache_path, "config.json")
    if os.path.exists(model_config_file):
        logger.info(f"Skip download model '{model_id}' due to config '{model_config_file}' exist")
        return
    
    subprocess.run(["mkdir", "-p", model_cache_path])
    subprocess.run(["mkdir", "-p", os.path.join(path, "refs")])
    
    if isS3 or isLocal:
        logger.info(f"Downloading '{model_id}' files from '{bucket_uri}' to '{model_cache_path}'")
    elif isGit:
        logger.info(f"Downloading '{model_id}' files from '{model_tmp_dir}' to '{model_cache_path}'")
    
    if isS3:
        if endpoint_url:
            subprocess.run([ "aws", "--endpoint-url", endpoint_url, "s3", "sync", "--quiet"] + s3_sync_args + [bucket_uri, model_cache_path])
        else:
            subprocess.run([ "aws", "s3", "sync", "--quiet"] + s3_sync_args + [bucket_uri, model_cache_path])
    elif isLocal:
        if not bucket_uri.endswith("/"):
            bucket_uri = bucket_uri + "/"
        subprocess.run(["cp -rf " + bucket_uri + "*" + " " + model_cache_path], shell=True)
    elif isGit:
        if not model_tmp_dir.endswith("/"):
            model_tmp_dir = model_tmp_dir + "/"
        subprocess.run(["cp -rf " + model_tmp_dir + "*" + " " + model_cache_path], shell=True)
        
    with open(os.path.join(path, "refs", "main"), "w") as f:
        f.write(f_hash)

def timeit(func):
    """
    Decorator to time a function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        start_time = time.monotonic()
        ret = func(*args, **kwargs)
        time_taken = time.monotonic() - start_time
        logger.info(f"LLM time counting fun {func} took {time_taken} s to complete")
        return ret

    return inner


def initialize_node(
    model_id: Optional[str] = None,
    s3_mirror_config: Optional[S3MirrorConfig] = None,
):
    """
    Performn initialization for a node.

    Currently, that means downloading the model from the S3 bucket.
    """
    # Create the torch cache kernels directory if it doesn't exist.
    # This is a workaround for a torch issue, where the kernels directory
    # cannot be created by torch if the parent directory doesn't exist.
    torch_cache_home = _get_torch_home()
    os.makedirs(os.path.join(torch_cache_home, "kernels"), exist_ok=True)

    if model_id and s3_mirror_config and (s3_mirror_config.bucket_uri or s3_mirror_config.git_uri):
        lock_path = os.path.expanduser(f"~/{model_id.replace('/', '--')}.lock")
        try:
            # Timeout 0 means there will be only one attempt to acquire
            # the file lock. If it cannot be aquired, a TimeoutError
            # will be thrown.
            # This allows us to make sure that subsequent processes don't
            # duplicate work.
            with FileLock(lock_path, timeout=0):
                endpoint_url = s3_mirror_config.endpoint_url
                bucket_uri = s3_mirror_config.bucket_uri
                git_uri = s3_mirror_config.git_uri
                s3_sync_args = s3_mirror_config.s3_sync_args
                try:
                    download_model(model_id, endpoint_url, bucket_uri, git_uri, s3_sync_args=s3_sync_args)
                    logger.info("Done downloading the model from bucket!")
                except RuntimeError:
                    logger.warning(
                        f"Unable to download the model from bucket. Traceback:\n {traceback.format_exc()}"
                    )
        except TimeoutError:
            # if the directory is already locked, then wait but do not do anything.
            with FileLock(lock_path, timeout=-1):
                pass


def merge_dicts(overwrite: dict, base: dict) -> dict:
    """
    Merge two dictionaries recursively, with keys from overwrite taking precedence.
    """
    base = base.copy()
    for key, value in overwrite.items():
        if isinstance(value, dict):
            # get node or create one
            node = base.setdefault(key, {})
            merge_dicts(value, node)
        else:
            base[key] = value

    return base


async def init_torch_dist_process_group_async(
    workers: List[ActorHandle],
    backend: str = "gloo",
    init_method: str = "env",
) -> List[int]:
    """Initialize a torch distributed process group asynchronously.

    This is identical to
    ``ray.air.util.torch_dist.init_torch_dist_process_group``
    but uses asyncio to avoid blocking the event loop.

    Note: this util assumes that the order of the workers passed in
    are their global ranks.

    Args:
        workers: A list of TorchDistributedWorker actors.
        backend: The torch distributed backend to use,
            possible choices are "gloo" or "nccl".
        init_method: The initialization method to use,
            possible choices are "env" or "tcp".

    Returns:
        Local ranks on their respective nodes for the list of workers.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed torch is not available.")

    # Build a map from node_id to workers on that node.
    node_and_gpu_ids = await asyncio.gather(
        *[w.execute.remote(_get_node_and_gpu_ids) for w in workers]
    )
    # All the workers on a specific node.
    node_to_workers = defaultdict(list)
    # All the gpu ids visible to all the workers on a specific node.
    node_to_gpu_ids = defaultdict(set)
    for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
        node_to_workers[node_id].append(i)
        # Force list.
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        # It is possible for a worker to have access to multiple GPUs.
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(gpu_id)

    # Assume the first worker is the master.
    master_addr, master_port = (
        await asyncio.gather(workers[0].execute.remote(get_address_and_port))
    )[0]

    setup_futures = []
    world_size = len(workers)
    local_ranks = []
    for rank, worker in enumerate(workers):
        node_id = node_and_gpu_ids[rank][0]
        local_rank = node_to_workers[node_id].index(rank)
        local_world_size = len(node_to_workers[node_id])
        setup_futures.append(
            worker.execute.remote(
                _init_torch_distributed,
                init_method=init_method,
                backend=backend,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                # list(set) will sort the gpu ids, so VISIBLE_CUDA_DEVICES
                # is always sorted.
                gpu_ids=list(node_to_gpu_ids[node_id]),
            )
        )
        local_ranks.append(local_rank)

    # Wait for all workers to join the process group.
    await asyncio.gather(*setup_futures)

    return local_ranks
