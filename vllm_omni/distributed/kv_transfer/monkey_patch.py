"""Monkey-patch vLLM's native ``MooncakeConnector`` with the patched version
that fixes request-ID mismatch in PD disaggregation.

Call :func:`apply_mooncake_connector_patch` at stage startup (before the
vLLM engine is constructed) so that vLLM's own ``MooncakeConnector``
reference resolves to our patched subclass.

The patching follows the same ``sys.modules`` iteration pattern used by
``vllm_omni/patch.py`` for other class replacements.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

_patched: bool = False


def apply_mooncake_connector_patch(engine_id: str | None = None) -> bool:
    """Replace vLLM's ``MooncakeConnector`` with the patched version.

    Parameters
    ----------
    engine_id:
        Optional engine identifier passed through to the patched class
        (used in log messages for debugging PD disaggregation).

    Returns
    -------
    bool
        ``True`` if the patch was applied (or was already applied),
        ``False`` if vLLM's MooncakeConnector could not be imported
        (e.g. vLLM not installed or version mismatch).
    """
    global _patched
    if _patched:
        logger.debug(
            "[monkey_patch] MooncakeConnector patch already applied, skipping"
        )
        return True

    # --- 0. Version compatibility check ----------------------------------
    _VLLM_MIN_VERSION = "0.8.0"
    try:
        import vllm
        if hasattr(vllm, "__version__") and vllm.__version__ < _VLLM_MIN_VERSION:
            logger.warning(
                "[monkey_patch] vLLM %s < %s — MooncakeConnector patch "
                "may be incompatible",
                vllm.__version__,
                _VLLM_MIN_VERSION,
            )
    except Exception:
        pass

    # --- 1. Import the original class -----------------------------------
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            mooncake_connector as _mc_module,
        )
        _OriginalMooncakeConnector = _mc_module.MooncakeConnector
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "[monkey_patch] Cannot import vLLM MooncakeConnector — "
            "patch NOT applied: %s",
            exc,
        )
        return False

    # --- 2. Build patched class -----------------------------------------
    from vllm_omni.distributed.kv_transfer.patched_mooncake_connector import (
        create_patched_mooncake_connector,
    )

    PatchedClass = create_patched_mooncake_connector(engine_id=engine_id)

    # --- 3. Replace in the defining module ------------------------------
    _mc_module.MooncakeConnector = PatchedClass
    logger.info(
        "[monkey_patch] Replaced MooncakeConnector in %s (engine_id=%s)",
        _mc_module.__name__,
        engine_id,
    )

    # --- 4. Replace in all already-imported modules ---------------------
    # Same pattern as vllm_omni/patch.py:18-33
    for module_name, module in sys.modules.items():
        if "vllm" not in module_name:
            continue
        if (
            hasattr(module, "MooncakeConnector")
            and module.MooncakeConnector is _OriginalMooncakeConnector
        ):
            module.MooncakeConnector = PatchedClass
            logger.debug(
                "[monkey_patch] Also patched MooncakeConnector in %s",
                module_name,
            )

    _patched = True
    logger.info("[monkey_patch] MooncakeConnector patch applied successfully")
    return True
