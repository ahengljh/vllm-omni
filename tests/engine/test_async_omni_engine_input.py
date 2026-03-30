from unittest.mock import Mock

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.model_executor.stage_input_processors.bagel import ExpandedPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine_core_request() -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id="req-1",
        prompt_token_ids=[1, 1, 1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def test_build_add_request_message_preserves_additional_information():
    engine = object.__new__(AsyncOmniEngine)
    params = SamplingParams(max_tokens=8)
    engine.default_sampling_params_list = [params]
    engine.stage_metadata = [{"stage_type": "llm"}]
    engine.supported_tasks = ("speech",)

    input_processor = Mock()
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor

    output_processor = Mock()
    engine.output_processors = [output_processor]

    prompt = {
        "prompt_token_ids": [1, 1, 1],
        "additional_information": {
            "text": ["hello world"],
            "speaker": ["vivian"],
        },
    }

    msg = engine._build_add_request_message(
        request_id="req-1",
        prompt=prompt,
        sampling_params_list=[params],
        final_stage_id=0,
        arrival_time=0.0,
    )

    request = msg["prompt"]
    assert isinstance(request, OmniEngineCoreRequest)
    assert request.external_req_id == "req-1"
    assert request.additional_information is not None
    assert request.additional_information.entries["text"].list_data == ["hello world"]
    assert request.additional_information.entries["speaker"].list_data == ["vivian"]
    output_processor.add_request.assert_called_once()


def test_build_cfg_companion_stage0_params_shortens_decode():
    params = SamplingParams(max_tokens=2048)
    params.min_tokens = 7
    params.stop = ["</s>"]
    params.stop_token_ids = [42]
    params.include_stop_str_in_output = True
    params.extra_args = {"negative_prompt": ""}

    companion = AsyncOmniEngine._build_cfg_companion_stage0_params(params)

    assert companion is not params
    assert companion.max_tokens == 1
    assert companion.min_tokens == 1
    assert companion.stop == []
    assert companion.stop_token_ids == []
    assert companion.include_stop_str_in_output is False
    assert companion.extra_args == {"negative_prompt": ""}

    assert params.max_tokens == 2048
    assert params.min_tokens == 7
    assert params.stop == ["</s>"]
    assert params.stop_token_ids == [42]
    assert params.include_stop_str_in_output is True
    assert params.extra_args == {"negative_prompt": ""}


def test_enqueue_cfg_companions_normalizes_split_img2img_prompt():
    engine = object.__new__(AsyncOmniEngine)

    renderer = Mock()
    renderer._tokenize_prompt.return_value = {"prompt_token_ids": [11, 22]}
    renderer.default_cmpl_tok_params = object()

    input_processor = Mock()
    input_processor.renderer = renderer
    input_processor.process_inputs.return_value = _make_engine_core_request()
    engine.input_processor = input_processor
    engine.output_processors = [Mock()]
    engine.request_queue = Mock()
    engine.request_queue.sync_q = Mock()
    engine.supported_tasks = ("generate",)
    engine.prompt_expand_func = Mock(
        return_value=[
            ExpandedPrompt(
                prompt={
                    "prompt": "<|fim_middle|>negative",
                    "modalities": ["img2img"],
                    "multi_modal_data": {"img2img": "image-bytes"},
                },
                role="cfg_text",
                request_id_suffix="__cfg_text",
            )
        ]
    )

    params = SamplingParams(max_tokens=32)
    params.extra_args = {"negative_prompt": ""}

    engine._enqueue_cfg_companions(
        parent_id="req-1",
        original_prompt={
            "prompt": "edit this image",
            "modalities": ["img2img"],
            "multi_modal_data": {"img2img": "image-bytes"},
        },
        stage0_params=params,
        sampling_params_list=[params],
    )

    processed_prompt = input_processor.process_inputs.call_args.kwargs["prompt"]
    assert processed_prompt["prompt"] == "<|fim_middle|><|fim_middle|>negative"
    assert processed_prompt["multi_modal_data"]["img2img"] == ["image-bytes", "image-bytes"]
    assert len(processed_prompt["multi_modal_uuids"]["img2img"]) == 2
    assert processed_prompt["prompt_token_ids"] == [11, 22]
