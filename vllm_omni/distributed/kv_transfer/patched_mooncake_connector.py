"""Patched MooncakeConnector that threads ``remote_request_id`` through
KV transfer params so the decode engine can look up the KV cache stored
by the prefill engine under its (different) internal request ID.

Usage::

    from vllm_omni.distributed.kv_transfer.patched_mooncake_connector import (
        create_patched_mooncake_connector,
    )

    PatchedCls = create_patched_mooncake_connector(engine_id="my-engine")
    # PatchedCls is a subclass of vLLM's MooncakeConnector
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Patched metadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatchedRecvReqMeta:
    """Extended receive-request metadata that carries the prefill engine's
    internal request ID (``remote_request_id``) alongside the local one.
    """
    request_id: str
    remote_request_id: str
    local_block_ids: list[int]
    kv_transfer_params: dict[str, Any]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_patched_mooncake_connector(engine_id: str | None = None):
    """Return a *subclass* of vLLM's ``MooncakeConnector`` with
    ``remote_request_id`` support baked in.

    The import is lazy so this module can be safely imported even when
    vLLM is not installed (e.g. during linting / light tests).

    Parameters
    ----------
    engine_id:
        Optional identifier for this engine instance (for logging).

    Returns
    -------
    type
        A class that is a proper subclass of
        ``vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector.MooncakeConnector``.
    """
    # Lazy import — the GPU environment has vLLM; CI / linting may not.
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_connector import (
        MooncakeConnector as _OriginalMooncakeConnector,
    )

    class PatchedMooncakeConnector(_OriginalMooncakeConnector):
        """MooncakeConnector subclass that fixes the request-ID mismatch
        in prefill-decode disaggregation.

        Key changes
        -----------
        * ``request_finished`` injects ``remote_request_id`` (the prefill
          engine's internal request ID) into ``kv_transfer_params`` so the
          orchestrator can forward it to the decode engine.
        * ``add_new_req`` uses ``remote_request_id`` from
          ``kv_transfer_params`` when ``load_remote_cache=True``, creating a
          ``PatchedRecvReqMeta`` instead of the default ``RecvReqMeta``.
        * ``group_kv_pull`` sends ZMQ requests using
          ``meta.remote_request_id``.
        * ``receive_kv`` maps the remote ID back to the local ID after
          transfer.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.engine_id: str | None = engine_id
            # remote_request_id → local_request_id mapping for in-flight pulls
            self.remote_to_local_req: dict[str, str] = {}
            logger.info(
                "[PatchedMooncakeConnector] Initialized (engine_id=%s)",
                self.engine_id,
            )

        # ---- prefill side: inject remote_request_id into output ----

        def request_finished(
            self,
            request: Any,
            block_ids: list[int],
        ) -> dict[str, Any] | None:
            """Call the original ``request_finished``, then patch the returned
            ``kv_transfer_params`` dict with ``remote_request_id``.

            The original implementation may store the request in
            ``_reqs_need_send`` as a ``(Request, list[int])`` tuple; we also
            normalise that to just ``list[int]`` to prevent downstream
            serialisation issues.
            """
            result = super().request_finished(request, block_ids)

            # --- normalise _reqs_need_send values -----------------------
            # The base class may store (Request, list[int]) tuples.  Down-
            # stream code that iterates over the dict values sometimes
            # expects bare list[int].  Normalise eagerly so we don't hit
            # "tuple is not subscriptable" errors later.
            req_id = getattr(request, "request_id", None)
            if req_id and hasattr(self, "_reqs_need_send"):
                entry = self._reqs_need_send.get(req_id)
                if isinstance(entry, tuple) and len(entry) == 2:
                    self._reqs_need_send[req_id] = entry[1]

            # --- inject remote_request_id into kv_transfer_params -------
            if result is not None and isinstance(result, dict):
                result["remote_request_id"] = req_id or "NOT_SET"
                # Ensure host/port are present for decode-side look-up
                if hasattr(self, "side_channel_host"):
                    result.setdefault("remote_host", self.side_channel_host)
                if hasattr(self, "side_channel_port"):
                    result.setdefault("remote_port", self.side_channel_port)
                logger.debug(
                    "[PatchedMooncakeConnector] request_finished: "
                    "req_id=%s remote_request_id=%s engine_id=%s",
                    req_id,
                    result.get("remote_request_id"),
                    self.engine_id,
                )

            return result

        # ---- decode side: use remote_request_id for look-up ----

        def add_new_req(
            self,
            request_id: str,
            local_block_ids: list[int],
            kv_transfer_params: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            """Override to store a ``PatchedRecvReqMeta`` that remembers the
            prefill engine's ``remote_request_id``.

            When ``kv_transfer_params`` contains ``"remote_request_id"``, we
            use it for the ZMQ look-up key.  Otherwise we fall back to the
            local ``request_id`` (original behaviour).
            """
            kv_transfer_params = kv_transfer_params or {}
            load_remote_cache = kv_transfer_params.get(
                "do_remote_prefill",
                kv_transfer_params.get("load_remote_cache", False),
            )

            if load_remote_cache:
                remote_request_id = kv_transfer_params.get(
                    "remote_request_id", request_id
                )
                meta = PatchedRecvReqMeta(
                    request_id=request_id,
                    remote_request_id=remote_request_id,
                    local_block_ids=local_block_ids,
                    kv_transfer_params=kv_transfer_params,
                )
                # Store in the same structure the base class uses
                if not hasattr(self, "_reqs_need_recv"):
                    self._reqs_need_recv = {}
                self._reqs_need_recv[request_id] = meta
                logger.debug(
                    "[PatchedMooncakeConnector] add_new_req (recv): "
                    "local_id=%s remote_id=%s engine_id=%s",
                    request_id,
                    remote_request_id,
                    self.engine_id,
                )
            else:
                # Producer side — delegate to original
                super().add_new_req(
                    request_id,
                    local_block_ids,
                    kv_transfer_params,
                    **kwargs,
                )

        def group_kv_pull(self, metadata: Any | None = None) -> None:
            """Override to use ``meta.remote_request_id`` as the ZMQ look-up
            key instead of the local request ID.

            After issuing the pull, we record the remote→local mapping in
            ``self.remote_to_local_req`` so ``receive_kv`` can translate
            back.
            """
            if not hasattr(self, "_reqs_need_recv") or not self._reqs_need_recv:
                return

            for local_id, meta in list(self._reqs_need_recv.items()):
                if isinstance(meta, PatchedRecvReqMeta):
                    remote_id = meta.remote_request_id
                    self.remote_to_local_req[remote_id] = local_id
                    logger.debug(
                        "[PatchedMooncakeConnector] group_kv_pull: "
                        "remote_id=%s -> local_id=%s",
                        remote_id,
                        local_id,
                    )
                    # Replace with a fake meta that uses remote_id as request_id
                    # so the base class ZMQ logic uses remote_id to look up KV
                    patched_meta = type(meta)(
                        request_id=remote_id,
                        remote_request_id=remote_id,
                        local_block_ids=meta.local_block_ids,
                        kv_transfer_params=meta.kv_transfer_params,
                    )
                    self._reqs_need_recv[local_id] = patched_meta

            # Delegate the actual ZMQ transfer to the base class
            super().group_kv_pull(metadata)

        def receive_kv(self, path: Any = None, req_blocks: Any = None) -> Any:
            """After the base class completes the ZMQ transfer, map
            ``remote_id`` back to ``local_id`` in any result structures.
            """
            result = super().receive_kv(path, req_blocks)

            # Clean up any completed remote→local mappings
            if self.remote_to_local_req:
                completed = []
                for remote_id, local_id in self.remote_to_local_req.items():
                    if not hasattr(self, "_reqs_need_recv") or local_id not in self._reqs_need_recv:
                        completed.append(remote_id)
                for remote_id in completed:
                    popped_local = self.remote_to_local_req.pop(remote_id, None)
                    logger.debug(
                        "[PatchedMooncakeConnector] receive_kv done: "
                        "remote_id=%s -> local_id=%s",
                        remote_id,
                        popped_local,
                    )

            return result

    # Preserve the original module name for isinstance checks in vLLM
    PatchedMooncakeConnector.__qualname__ = "MooncakeConnector"

    return PatchedMooncakeConnector
