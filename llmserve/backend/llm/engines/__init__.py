from typing import TYPE_CHECKING, Type
from llmserve.backend.logger import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:
    from ._base import LLMEngine

from .generic import GenericEngine
try:
    from .vllm import VllmEngine
except ImportError:
    logger.info("Import vllm related stuff failed, please make sure 'vllm' is installed.")

def get_engine_cls_by_name(name: str) -> Type["LLMEngine"]:
    lowercase_globals = {k.lower(): v for k, v in globals().items()}
    ret = lowercase_globals.get(
        f"{name.lower()}engine", lowercase_globals.get(name.lower(), None)
    )
    assert ret
    return ret


__all__ = [
    "get_engine_cls_by_name",
    "GenericEngine",
    "VllmEngine",
]
