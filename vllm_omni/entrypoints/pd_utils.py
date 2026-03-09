# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PD (Prefill-Decode) disaggregation helpers.

Extracted from ``OmniBase`` so that ``omni.py`` stays focused on
orchestration logic.  ``OmniBase`` inherits from ``PDDisaggregationMixin``
which keeps all ``self._method()`` call-sites unchanged.
"""

import logging
import threading
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni_stage import OmniStage

# Default port for Mooncake KV transfer bootstrap service.
# Used when ``mooncake_bootstrap_port`` is not set in kv_connector_extra_config.
_DEFAULT_MOONCAKE_BOOTSTRAP_PORT = 25201

logger = logging.getLogger(__name__)


class PDDisaggregationMixin:
    """Mixin supplying PD disaggregation helpers to ``OmniBase``."""

    # ------------------------------------------------------------------
    # PD (Prefill-Decode) disaggregation helpers
    # ------------------------------------------------------------------

    def _init_pd_state(self) -> None:
        """Initialise PD disaggregation state attributes.

        Called from ``OmniBase.__init__`` after ``_initialize_stages``.
        """
        self._pd_separation_pair: tuple[int, int] | None = self._detect_pd_separation()
        self._pd_connector_info: dict[str, Any] | None = None
        self._pd_kv_params_by_req: dict[str, dict[str, Any]] = {}
        # Lock protects _pd_kv_params_by_req for the async path (AsyncOmni)
        # where store and pop may run from different coroutines.  In the sync
        # path (_run_generation) store and pop happen sequentially in the same
        # thread, but the lock is harmless and keeps the code uniform.
        self._pd_kv_params_lock = threading.Lock()
        if self._pd_separation_pair is not None:
            self._validate_pd_separation_config()
            self._pd_connector_info = self._get_pd_connector_info()
            p_id, d_id = self._pd_separation_pair
            logger.info(
                "[%s] PD disaggregation detected: prefill=stage-%d, decode=stage-%d",
                self._name,
                p_id,
                d_id,
            )

    def _detect_pd_separation(self) -> tuple[int, int] | None:
        """Scan stage_list for a prefill-only / decode-only pair.

        Returns:
            ``(prefill_stage_id, decode_stage_id)`` if found, else ``None``.

        Raises:
            ValueError: If multiple PD pairs are detected (not supported).
        """
        # Single pass: collect prefill stages keyed by both list index and
        # stage_id so decode stages can match against either.
        prefill_by_id: dict[int, int] = {}  # stage_id_or_index → list index
        decode_indices: list[int] = []
        for i, stage in enumerate(self.stage_list):
            if getattr(stage, "is_prefill_only", False):
                prefill_by_id[i] = i
                sid = getattr(stage, "stage_id", i)
                if sid != i:
                    prefill_by_id[sid] = i
            if getattr(stage, "is_decode_only", False):
                decode_indices.append(i)

        # Match decode stages to prefill stages via engine_input_source.
        # This is O(d*s) where d = number of decode stages (typically 1)
        # and s = number of source IDs per decode stage (typically 1..2).
        pd_pairs: list[tuple[int, int]] = []
        for j in decode_indices:
            source_ids = getattr(self.stage_list[j], "engine_input_source", [])
            for src in source_ids:
                if src in prefill_by_id:
                    pd_pairs.append((prefill_by_id[src], j))
                    break

        if len(pd_pairs) > 1:
            raise ValueError(
                f"Multiple PD pairs detected ({pd_pairs}); only a single PD pair per pipeline is supported"
            )
        return pd_pairs[0] if pd_pairs else None

    @staticmethod
    def _to_dict(obj: Any, default: Any = None) -> dict[str, Any] | None:
        """Convert *obj* to a plain ``dict``, trying several strategies.

        Returns *default* when *obj* is ``None`` or conversion fails.
        Typical usage::

            self._to_dict(kv_cfg, default={})   # replaces _kv_cfg_to_dict
            self._to_dict(kv_params)             # replaces _normalize_kv_transfer_params
        """
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj
        if is_dataclass(obj):
            try:
                return asdict(obj)
            except Exception:
                return default
        for attr in ("model_dump", "dict"):
            if hasattr(obj, attr):
                try:
                    return getattr(obj, attr)()
                except Exception:
                    pass
        if hasattr(obj, "items"):
            try:
                return dict(obj)
            except Exception:
                pass
        try:
            return dict(obj)
        except Exception:
            try:
                return vars(obj)
            except Exception:
                logger.debug(
                    "Unable to convert object of type %s to dict",
                    type(obj),
                )
                return default

    # Intentional thin wrappers over _to_dict with different defaults:
    # _kv_cfg_to_dict returns {} (never None) for safe .get() chains.
    # _normalize_kv_transfer_params returns None when absent (caller checks).
    def _kv_cfg_to_dict(self, kv_cfg: Any) -> dict[str, Any]:
        return self._to_dict(kv_cfg, default={}) or {}

    def _normalize_kv_transfer_params(self, kv_params: Any) -> dict[str, Any] | None:
        return self._to_dict(kv_params)

    def _validate_pd_separation_config(self) -> None:
        """Validate that PD stage configurations are consistent."""
        assert self._pd_separation_pair is not None
        p_id, d_id = self._pd_separation_pair
        p_stage = self.stage_list[p_id]
        d_stage = self.stage_list[d_id]

        def _get_kv_cfg(stage: "OmniStage") -> dict[str, Any]:
            ea = stage.engine_args
            cfg = getattr(ea, "kv_transfer_config", None)
            if cfg is None:
                cfg = ea.get("kv_transfer_config", None) if hasattr(ea, "get") else None
            if cfg is None:
                raise ValueError(
                    f"Stage-{stage.stage_id} is marked for PD disaggregation "
                    "but has no 'kv_transfer_config' in engine_args"
                )
            cfg_dict = self._kv_cfg_to_dict(cfg)
            if not cfg_dict:
                raise ValueError(
                    f"Stage-{stage.stage_id} kv_transfer_config ({type(cfg).__name__}) could not be parsed into a dict"
                )
            return cfg_dict

        p_cfg = _get_kv_cfg(p_stage)
        d_cfg = _get_kv_cfg(d_stage)

        p_role = p_cfg.get("kv_role")
        d_role = d_cfg.get("kv_role")
        if p_role not in ("kv_producer", "kv_both"):
            raise ValueError(f"Prefill stage-{p_id} kv_role must be 'kv_producer' or 'kv_both', got '{p_role}'")
        if d_role not in ("kv_consumer", "kv_both"):
            raise ValueError(f"Decode stage-{d_id} kv_role must be 'kv_consumer' or 'kv_both', got '{d_role}'")

        d_sources = list(getattr(d_stage, "engine_input_source", []) or [])
        if p_id not in d_sources and p_stage.stage_id not in d_sources:
            raise ValueError(f"Decode stage-{d_id} must list prefill stage-{p_id} in engine_input_source")

        p_conn = p_cfg.get("kv_connector")
        d_conn = d_cfg.get("kv_connector")
        if p_conn != d_conn:
            raise ValueError(f"PD connector mismatch: prefill uses '{p_conn}', decode uses '{d_conn}'")
        if not p_conn:
            raise ValueError("PD disaggregation requires kv_connector to be set in kv_transfer_config")

        for key in ("kv_buffer_device", "kv_buffer_size"):
            p_val = p_cfg.get(key)
            d_val = d_cfg.get(key)
            if p_val is not None and d_val is not None and p_val != d_val:
                raise ValueError(f"PD {key} mismatch: prefill uses '{p_val}', decode uses '{d_val}'")

        # Validate tensor_parallel_size matches between prefill and decode
        p_tp = getattr(getattr(p_stage, "engine_args", None), "tensor_parallel_size", 1)
        d_tp = getattr(getattr(d_stage, "engine_args", None), "tensor_parallel_size", 1)
        if p_tp != d_tp:
            raise ValueError(f"PD stages must have matching tensor_parallel_size: prefill={p_tp}, decode={d_tp}")

    def _get_pd_connector_info(self) -> dict[str, Any] | None:
        """Extract prefill engine KV connector info from stage config."""
        if self._pd_separation_pair is None:
            return None

        p_id, _ = self._pd_separation_pair
        p_stage = self.stage_list[p_id]

        ea = p_stage.engine_args
        kv_cfg = getattr(ea, "kv_transfer_config", None)
        if kv_cfg is None and hasattr(ea, "get"):
            kv_cfg = ea.get("kv_transfer_config")
        if kv_cfg is None:
            return None

        kv_cfg_dict = self._kv_cfg_to_dict(kv_cfg)
        if not kv_cfg_dict:
            return None

        engine_id = kv_cfg_dict.get("engine_id")
        kv_connector = str(kv_cfg_dict.get("kv_connector", "") or "")
        extra_cfg = kv_cfg_dict.get("kv_connector_extra_config", {}) or {}
        if not isinstance(extra_cfg, dict):
            extra_cfg = self._kv_cfg_to_dict(extra_cfg)

        info: dict[str, Any] = {"prefill_engine_id": engine_id}

        if "mooncake" in kv_connector.lower():
            bootstrap_port = extra_cfg.get("mooncake_bootstrap_port", None)
            if bootstrap_port is None:
                bootstrap_port = _DEFAULT_MOONCAKE_BOOTSTRAP_PORT
            kv_ip = kv_cfg_dict.get("kv_ip") or "127.0.0.1"
            info["prefill_bootstrap_addr"] = f"{kv_ip}:{bootstrap_port}"

        logger.debug("[%s] PD connector info: %s", self._name, info)
        return info

    def _prepare_prefill_sampling_params(self, req_id: str, sp: "SamplingParams") -> "SamplingParams":
        sp = sp.clone()
        sp.max_tokens = 1
        if hasattr(sp, "min_tokens"):
            try:
                sp.min_tokens = 1
            except Exception:
                pass
        # Neutralize stop conditions so the prefill always finishes with
        # finish_reason='length' (not 'stop').  MooncakeConnector cancels
        # KV transfer for any reason other than FINISHED_LENGTH_CAPPED.
        sp.stop = []
        sp.stop_token_ids = []
        sp.include_stop_str_in_output = False
        if sp.extra_args is None:
            sp.extra_args = {}
        kv_params = self._normalize_kv_transfer_params(sp.extra_args.get("kv_transfer_params"))
        merged: dict[str, Any] = {}
        if kv_params:
            merged.update(kv_params)
        merged.update(
            {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": f"xfer-{req_id}",
            }
        )
        sp.extra_args["kv_transfer_params"] = merged
        logger.debug(
            "[PD] _prepare_prefill_sampling_params: req=%s max_tokens=%s kv_transfer_params=%s extra_args_id=%s",
            req_id,
            sp.max_tokens,
            merged,
            id(sp.extra_args),
        )
        return sp

    def _pop_pd_kv_params(self, req_id: str, fallback: Any | None = None) -> dict[str, Any] | None:
        kv_params = self._normalize_kv_transfer_params(fallback)
        with self._pd_kv_params_lock:
            stored = self._pd_kv_params_by_req.pop(req_id, None)
        if kv_params is None:
            kv_params = stored
        return kv_params

    def _drop_pd_kv_params(self, req_id: str) -> None:
        with self._pd_kv_params_lock:
            self._pd_kv_params_by_req.pop(req_id, None)

    def _extract_kv_transfer_params(self, engine_outputs: Any) -> dict[str, Any] | None:
        """Extract kv_transfer_params from already-loaded engine outputs.

        Called after engine outputs have been deserialized from IPC so that
        shared memory is only read once.

        Note: Whether MooncakeConnector propagates kv_transfer_params back
        via EngineCoreOutput depends on the vLLM version.  Some versions
        return ``None`` from ``request_finished()`` while others return
        actual params (remote_host, remote_port, etc.).  When available,
        these are merged into the decode-side kv_transfer_params.
        """
        outputs = engine_outputs if isinstance(engine_outputs, list) else [engine_outputs]
        for output in outputs:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                logger.debug(
                    "[PD] Extracted kv_transfer_params from engine output: %s",
                    kv_params,
                )
                return self._normalize_kv_transfer_params(kv_params)
        return None
