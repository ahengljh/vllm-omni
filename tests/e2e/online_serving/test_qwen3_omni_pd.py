"""
E2E online serving tests for Qwen3-Omni-MoE with PD (Prefill-Decode) disaggregation.

Tests both text-only and audio output modalities via the OpenAI-compatible API
through the 4-stage PD pipeline: Prefill -> Decode -> Talker -> Code2Wav.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import (
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

# PD disaggregation CI stage config (requires 3x GPUs)
stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_pd_ci.yaml")]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=3)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_pd_text_to_text(omni_server, openai_client) -> None:
    """
    Test PD disaggregation with text-only output via OpenAI API.
    Deploy Setting: PD separation yaml
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_only"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=3)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_pd_mix_to_text_audio(omni_server, openai_client) -> None:
    """
    Test PD disaggregation with multi-modal input and text+audio output via OpenAI API.
    Deploy Setting: PD separation yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {
            "audio": ["water", "chirping", "crackling", "rain"],
            "image": ["square", "quadrate"],
        },
    }

    openai_client.send_request(request_config)
