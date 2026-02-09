from collections.abc import Callable
from typing import Any

import cloudpickle
from pydantic import ValidationError
from tqdm import tqdm

# External library imports (vLLM)
from vllm.config import CompilationConfig, StructuredOutputsConfig, is_init_field
from vllm.entrypoints.llm import LLM
from vllm.logger import init_logger
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.usage.usage_lib import UsageContext
from vllm.utils.counter import Counter
from vllm.v1.engine.llm_engine import LLMEngine

from vllm_omni.distributed.omni_connectors import initialize_orchestrator_connectors

# Internal imports (our code)
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.input_processor import OmniInputProcessor
from vllm_omni.engine.output_processor import MultimodalOutputProcessor
from vllm_omni.entrypoints.utils import (
    load_stage_configs_from_model,
    load_stage_configs_from_yaml,
    resolve_model_config_path,
)

logger = init_logger(__name__)


class OmniLLM(LLM):
    """Main entry point for vLLM-Omni inference.

    This class extends the base vLLM LLM class with omni-specific
    processors for handling multimodal inputs and outputs. It provides
    configuration loading for multi-stage pipelines, while stage management
    is handled by the Omni class.

    Args:
        model: Model name or path to load
        stage_configs_path: Optional path to YAML file containing stage
            configurations. If None, configurations are loaded from the model.
        log_stats: Whether to enable statistics logging
        compilation_config: Optional compilation configuration. Can be an
            integer (compilation level), dict, or CompilationConfig instance.
        hf_overrides: Optional HuggingFace model configuration overrides
        structured_outputs_config: Optional structured outputs configuration.
            Can be a dict or StructuredOutputsConfig instance.
        init_sleep_seconds: Number of seconds to sleep between starting
            each stage process during initialization (used by Omni class)
        shm_threshold_bytes: Threshold in bytes for using shared memory
            for IPC. Objects larger than this threshold will use shared memory.
        batch_timeout: Timeout in seconds for batching requests within a stage
        init_timeout: Timeout in seconds for waiting for all stages to initialize
        **kwargs: Additional keyword arguments passed to the base LLM class
            and engine

    Example:
        >>> llm = OmniLLM(model="Qwen/Qwen2.5-Omni-7B")
        >>> # Stage management is handled by Omni class
    """

    def __init__(
        self,
        model: str,
        stage_configs_path: str | None = None,
        log_stats: bool = False,
        compilation_config: int | dict[str, Any] | CompilationConfig | None = None,
        hf_overrides: dict[str, Any] | None = None,
        structured_outputs_config: dict[str, Any] | StructuredOutputsConfig | None = None,
        init_sleep_seconds: int = 20,
        shm_threshold_bytes: int = 65536,
        batch_timeout: int = 10,
        init_timeout: int = 300,
        **kwargs: Any,
    ):
        """LLM constructor with omni-specific configuration loading."""
        # Store stage management parameters (used by Omni class)
        self.worker_backend = kwargs.get("worker_backend", "multi_process")
        self.ray_address = kwargs.get("ray_address", None)
        self.batch_timeout = batch_timeout
        self.log_stats: bool = bool(log_stats)

        # Load stage configurations
        if stage_configs_path is None:
            self.config_path = resolve_model_config_path(model)
            self.stage_configs = load_stage_configs_from_model(model)
        else:
            self.config_path = stage_configs_path
            self.stage_configs = load_stage_configs_from_yaml(stage_configs_path)

        # Initialize connectors
        self.omni_transfer_config, self.connectors = initialize_orchestrator_connectors(
            self.config_path, worker_backend=self.worker_backend, shm_threshold_bytes=shm_threshold_bytes
        )

        # Initialize LLM engine
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            # if the worker_cls is not qualified string name,
            # we serialize it using cloudpickle to avoid pickling issues
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig

            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(**raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict,
                    e,
                )
                raise ValueError(f"Invalid 'kv_transfer_config' provided: {e}") from e

        # Extract omni_kv_config from kwargs if present (injected by Omni)
        omni_kv_config = kwargs.pop("omni_kv_config", None)

        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(level=compilation_config)
            elif isinstance(compilation_config, dict):
                compilation_config_instance = CompilationConfig(
                    **{k: v for k, v in compilation_config.items() if is_init_field(CompilationConfig, k)}
                )
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()

        if structured_outputs_config is not None:
            if isinstance(structured_outputs_config, dict):
                structured_outputs_instance = StructuredOutputsConfig(
                    **{k: v for k, v in structured_outputs_config.items() if is_init_field(StructuredOutputsConfig, k)}
                )
            else:
                structured_outputs_instance = structured_outputs_config
        else:
            structured_outputs_instance = StructuredOutputsConfig()

        engine_args = OmniEngineArgs(
            model=model,
            compilation_config=compilation_config_instance,
            structured_outputs_config=structured_outputs_instance,
            omni_kv_config=omni_kv_config,
            **kwargs,
        )

        # Create the Engine (autoselects V0 vs V1)
        self.llm_engine = LLMEngine.from_engine_args(engine_args=engine_args, usage_context=UsageContext.LLM_CLASS)
        self.llm_engine.output_processor = MultimodalOutputProcessor(
            tokenizer=self.llm_engine.tokenizer,
            log_stats=self.llm_engine.log_stats,
            engine_core_output_type=engine_args.engine_output_type,
        )
        self.llm_engine.input_processor = OmniInputProcessor(vllm_config=self.llm_engine.vllm_config)
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        self.default_sampling_params: dict[str, Any] | None = None

        supported_tasks = self.llm_engine.get_supported_tasks()  # type: ignore

        logger.info("Supported_tasks: %s", supported_tasks)

        self.supported_tasks = supported_tasks

        # Load the Input/Output processor plugin if any
        io_processor_plugin = self.llm_engine.model_config.io_processor_plugin
        self.io_processor = get_io_processor(self.llm_engine.vllm_config, io_processor_plugin)
        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor

    def close(self) -> None:
        """Close resources.

        Note: Stage management is now handled by Omni class.
        This method closes the LLM engine but not stages.
        """
        # Close the LLM engine if it exists
        if hasattr(self, "llm_engine") and self.llm_engine is not None:
            if hasattr(self.llm_engine, "shutdown"):
                self.llm_engine.shutdown()

    def __del__(self) -> None:  # best-effort
        try:
            self.close()
        except Exception as e:
            logger.debug("[Orchestrator] __del__ close() raised: %s", e, exc_info=True)

    # ------------------------------------------------------------------
    # KV connector diagnostics (PD disaggregation)
    # ------------------------------------------------------------------

    def _get_kv_connector_scheduler(self):
        """Return the MooncakeConnectorScheduler instance if present."""
        try:
            engine_core = getattr(self.llm_engine, "engine_core", None)
            if engine_core is None:
                return None
            scheduler = getattr(engine_core, "scheduler", None)
            if scheduler is None:
                return None
            connector = getattr(scheduler, "connector", None)
            if connector is None:
                return None
            return getattr(connector, "connector_scheduler", None)
        except Exception:
            return None

    def _log_kv_connector_state(self, label: str) -> None:
        """Log complete KV connector scheduler state for PD debugging."""
        cs = self._get_kv_connector_scheduler()
        if cs is None:
            return

        sends = getattr(cs, "_reqs_need_send", {})
        recvs = getattr(cs, "_reqs_need_recv", {})
        not_proc = getattr(cs, "_reqs_not_processed", set())
        is_producer = getattr(cs, "is_kv_producer", None)
        is_consumer = getattr(cs, "is_kv_consumer", None)

        send_info = {}
        for req_id, val in sends.items():
            if isinstance(val, tuple) and len(val) == 2:
                req_obj, block_ids = val
                params = getattr(req_obj, "kv_transfer_params", None) or {}
                send_info[req_id] = {
                    "n_blocks": len(block_ids) if block_ids else 0,
                    "has_blocks": bool(block_ids),
                    "transfer_id": params.get("transfer_id"),
                }
            else:
                send_info[req_id] = repr(type(val))

        logger.warning(
            "[OmniLLM][KV-DIAG] %s: role=%s, "
            "sends=%d %s, recvs=%d, not_processed=%s",
            label,
            "producer" if is_producer else ("consumer" if is_consumer else "?"),
            len(sends),
            send_info if send_info else "{}",
            len(recvs),
            list(not_proc) if not_proc else "[]",
        )

    # ------------------------------------------------------------------

    def _run_engine(self, *, use_tqdm: bool | Callable[..., tqdm] = True) -> list[RequestOutput | PoolingRequestOutput]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            tqdm_func = use_tqdm if callable(use_tqdm) else tqdm
            pbar = tqdm_func(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, output: {0:.2f} toks/s"),
            )

        # PD diagnostic: check if engine has KV connector
        _kv_cs = self._get_kv_connector_scheduler()
        _has_kv = _kv_cs is not None
        if _has_kv:
            self._log_kv_connector_state("engine_loop_start")

        # Run the engine.
        outputs: list[RequestOutput | PoolingRequestOutput] = []
        total_in_toks = 0
        total_out_toks = 0
        _step = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            _step += 1
            _finished_this_step = False

            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    _finished_this_step = True

                    # PD diagnostic: log finished request details
                    if _has_kv:
                        _fr = (
                            output.outputs[0].finish_reason
                            if hasattr(output, "outputs") and output.outputs
                            else None
                        )
                        logger.warning(
                            "[OmniLLM][KV-DIAG] step %d: finished req=%s, "
                            "finish_reason=%s, kv_transfer_params=%s",
                            _step,
                            output.request_id,
                            _fr,
                            getattr(output, "kv_transfer_params", None),
                        )

                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            n = len(output.outputs)
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids) * n
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(len(stp.token_ids) for stp in output.outputs)
                            out_spd = total_out_toks / pbar.format_dict["elapsed"]
                            pbar.postfix = f"est. speed input: {in_spd:.2f} toks/s, output: {out_spd:.2f} toks/s"
                            pbar.update(n)
                        else:
                            pbar.update(1)
                        if pbar.n == num_requests:
                            pbar.refresh()

            if _has_kv and _finished_this_step:
                self._log_kv_connector_state(f"after_step_{_step}")

        if use_tqdm:
            pbar.close()
        # Sort the outputs by the int part of request ID which is in format of 'int-uuid'.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.

        # PD disaggregation: flush any pending KV connector sends that were
        # added by request_finished() after the last build_connector_meta()
        # call.  Without this flush, the prefill engine's worker never
        # receives the block IDs needed for KV transfer to the decode engine.
        if _has_kv:
            self._log_kv_connector_state("before_flush")
        self._flush_kv_connector_sends()
        if _has_kv:
            self._log_kv_connector_state("after_flush")

        return sorted(outputs, key=lambda x: int(x.request_id.split("-")[0]))

    def _flush_kv_connector_sends(self) -> None:
        """Flush pending KV connector send metadata to workers.

        When _run_engine() finishes a batch, request_finished() may have
        added entries to _reqs_need_send *after* the last
        build_connector_meta() call within that step's schedule().  In
        standard vLLM online serving this is not a problem because the
        engine loop continues and the next schedule() picks them up.  In
        OmniLLM batch mode the loop exits immediately, so we must run one
        more empty-request step to deliver the metadata.
        """
        try:
            engine_core = getattr(self.llm_engine, "engine_core", None)
            if engine_core is None:
                return
            scheduler = getattr(engine_core, "scheduler", None)
            if scheduler is None:
                return
            connector = getattr(scheduler, "connector", None)
            if connector is None:
                return

            # Check whether the scheduler-side connector has unflushed sends.
            cs = getattr(connector, "connector_scheduler", None)
            if cs is None:
                logger.warning(
                    "[OmniLLM][KV-DIAG] flush: connector exists but "
                    "connector_scheduler is None"
                )
                return

            pending = getattr(cs, "_reqs_need_send", None)
            not_proc = getattr(cs, "_reqs_not_processed", set())

            logger.warning(
                "[OmniLLM][KV-DIAG] flush: pending_sends=%d, "
                "not_processed=%d (%s), is_producer=%s",
                len(pending) if pending else 0,
                len(not_proc),
                list(not_proc) if not_proc else "[]",
                getattr(cs, "is_kv_producer", None),
            )

            if not pending:
                logger.warning(
                    "[OmniLLM][KV-DIAG] flush: _reqs_need_send is empty — "
                    "request_finished() likely did NOT re-add the entry "
                    "(build_connector_meta already consumed it with empty "
                    "block_ids during the same step's schedule())"
                )
                return

            # Log details of what we're about to flush
            for req_id, val in pending.items():
                if isinstance(val, tuple) and len(val) == 2:
                    _, block_ids = val
                    logger.warning(
                        "[OmniLLM][KV-DIAG] flush: will send req=%s, "
                        "block_ids=%s (n=%d)",
                        req_id,
                        block_ids[:8] if block_ids else "[]",
                        len(block_ids) if block_ids else 0,
                    )

            from vllm.v1.core.sched.output import SchedulerOutput

            # Create an empty scheduler output and attach connector metadata.
            so = SchedulerOutput.make_empty()
            meta = connector.build_connector_meta(so)
            so.kv_connector_metadata = meta

            # Log what the metadata contains
            if meta is not None:
                reqs_to_send = getattr(meta, "reqs_to_send", None)
                reqs_to_recv = getattr(meta, "reqs_to_recv", None)
                logger.warning(
                    "[OmniLLM][KV-DIAG] flush metadata: "
                    "reqs_to_send=%s, reqs_to_recv=%s",
                    {k: (tid, len(bids) if bids else 0)
                     for k, (tid, bids) in reqs_to_send.items()}
                    if reqs_to_send else "{}",
                    list(reqs_to_recv.keys()) if reqs_to_recv else "{}",
                )

            # Run an empty model step: the worker sees
            # total_num_scheduled_tokens == 0 and takes the no_forward()
            # path, which only processes connector metadata
            # (record_send_reqs → sets ready event on SendBlockMeta).
            model_executor = getattr(engine_core, "model_executor", None)
            if model_executor is None:
                return
            model_executor.execute_model(so)

            logger.warning("[OmniLLM] KV connector sends flushed successfully")
        except Exception:
            logger.warning(
                "[OmniLLM] Failed to flush KV connector sends",
                exc_info=True,
            )
