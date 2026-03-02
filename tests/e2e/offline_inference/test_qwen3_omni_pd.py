"""
E2E offline tests for Qwen3-Omni-MoE with PD (Prefill-Decode) disaggregation.

Tests both text-only and audio output modalities through the 4-stage
PD pipeline: Prefill -> Decode -> Talker -> Code2Wav.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_video,
)
from tests.utils import hardware_test

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# PD disaggregation CI stage config (requires 3x GPUs)
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_pd_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
        "text": "What is the capital of China? Answer in 20 words.",
    }
    return prompts.get(prompt_type, prompts["video"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=3)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_pd_text_only(omni_runner, omni_runner_handler) -> None:
    """Test PD disaggregation with text-only output (no talker/code2wav)."""
    request_config = {
        "prompts": get_question("text"),
        "modalities": ["text"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=3)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_pd_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test PD disaggregation with video input and audio output
    through the full 4-stage pipeline."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {
        "prompts": get_question("video"),
        "videos": video,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_request(request_config)
