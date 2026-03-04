"""Monkey-patch vLLM's ``MooncakeConnector`` with the patched version
that fixes request-ID mismatch in PD disaggregation.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_patched: bool = False


def _import_mooncake_module():
    """Import MooncakeConnector module, supporting both vLLM >=0.16 and older."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector
        return mooncake_connector
    except ImportError:
        pass
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import mooncake_connector
        return mooncake_connector
    except ImportError:
        return None


def apply_mooncake_connector_patch(engine_id: str | None = None) -> bool:
    """Replace vLLM's ``MooncakeConnector`` with the patched version."""
    global _patched
    if _patched:
        return True

    _mc_module = _import_mooncake_module()
    if _mc_module is None:
        logger.warning("[monkey_patch] Cannot import vLLM MooncakeConnector — patch NOT applied.")
        return False

    _OriginalClass = _mc_module.MooncakeConnector

    from vllm_omni.distributed.kv_transfer.patched_mooncake_connector import (
        create_patched_mooncake_connector,
    )
    PatchedClass = create_patched_mooncake_connector(engine_id=engine_id)

    _mc_module.MooncakeConnector = PatchedClass
    for _, module in sys.modules.items():
        if hasattr(module, "MooncakeConnector") and module.MooncakeConnector is _OriginalClass:
            module.MooncakeConnector = PatchedClass

    _patched = True
    logger.info("[monkey_patch] MooncakeConnector patch applied (engine_id=%s)", engine_id)
    return True
