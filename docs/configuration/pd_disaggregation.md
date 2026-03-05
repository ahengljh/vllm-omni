# Prefill-Decode (PD) Disaggregation

This guide explains how to enable Prefill-Decode disaggregation for Qwen3-Omni-MoE,
which splits the thinker stage into separate prefill and decode instances for
improved throughput and resource utilization.

## Overview

In the standard 3-stage pipeline, a single thinker engine handles both prompt
processing (prefill) and token generation (decode):

```
Thinker (prefill + decode) -> Talker -> Code2Wav
```

With PD disaggregation, the thinker is split into a 4-stage pipeline:

```
Stage 0: Thinker Prefill (KV producer)  --[KV cache via RDMA]--> Stage 1: Thinker Decode (KV consumer)
                                                                           |
                                                                    Stage 2: Talker
                                                                           |
                                                                    Stage 3: Code2Wav
```

The prefill stage processes the input prompt and transfers its KV cache to the
decode stage via vLLM's KV connector (e.g., MooncakeConnector). The decode stage
loads the KV cache and generates tokens autoregressively.

## Prerequisites

- **Hardware**: Minimum 3x GPUs (e.g., H100-80G). One GPU each for prefill and
  decode, one shared by talker and code2wav.
- **KV connector**: A supported KV transfer backend must be installed.
  Currently supported: MooncakeConnector (requires `mooncake-transfer-engine`).
- **Same TP size**: MooncakeConnector does not support heterogeneous tensor
  parallel sizes. Both prefill and decode stages must use the same
  `tensor_parallel_size`.

## Stage Config Files

Two PD stage configs are provided:

| Config File | Inter-Stage Connector | Use Case |
| :--- | :--- | :--- |
| `qwen3_omni_moe_pd_separation.yaml` | SharedMemoryConnector | Single-node deployment |
| `qwen3_omni_moe_pd_multiconnector.yaml` | MooncakeStoreConnector | Multi-node or Mooncake-based deployment |

Both use `MooncakeConnector` for KV cache transfer between prefill and decode.
The difference is in how intermediate data (embeddings, hidden states) is passed
between stages.

## GPU Layout

### `qwen3_omni_moe_pd_separation.yaml` (3 GPUs)

| GPU | Stage |
| :--- | :--- |
| GPU 0 | Stage 0: Thinker Prefill |
| GPU 1 | Stage 1: Thinker Decode |
| GPU 2 | Stage 2: Talker + Stage 3: Code2Wav |

### `qwen3_omni_moe_pd_multiconnector.yaml` (4 GPUs)

| GPU | Stage |
| :--- | :--- |
| GPU 0 | Reserved (e.g., API server) |
| GPU 1 | Stage 0: Thinker Prefill |
| GPU 2 | Stage 1: Thinker Decode |
| GPU 3 | Stage 2: Talker + Stage 3: Code2Wav |

Adjust `runtime.devices` in the YAML to match your GPU topology.

## Usage

### Offline Inference

```python
from vllm_omni import OmniLLM

model_name = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Single-node with SharedMemory
omni = OmniLLM(
    model=model_name,
    stage_configs_path="vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_separation.yaml",
)

# Or with MooncakeStoreConnector
omni = OmniLLM(
    model=model_name,
    stage_configs_path="vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_multiconnector.yaml",
)
```

### Online Serving

```bash
# Single-node with SharedMemory
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_separation.yaml

# With MooncakeStoreConnector
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_multiconnector.yaml
```

## Key Configuration Fields

### PD-Specific Stage Fields

```yaml
stage_args:
  - stage_id: 0
    is_prefill_only: true    # Mark as prefill-only stage
    # ...

  - stage_id: 1
    is_decode_only: true     # Mark as decode-only stage
    # ...
```

- `is_prefill_only: true` — The orchestrator overrides `max_tokens=1` so this
  stage only processes the prompt and saves KV cache. No autoregressive decoding.
- `is_decode_only: true` — This stage loads KV cache from the prefill stage and
  performs autoregressive token generation.

### KV Transfer Config

Both prefill and decode stages need `kv_transfer_config`:

```yaml
# Prefill stage (KV producer)
kv_transfer_config:
  kv_connector: "MooncakeConnector"
  kv_role: "kv_producer"
  kv_rank: 0
  kv_parallel_size: 2
  engine_id: "omni-thinker-prefill"
  kv_connector_extra_config:
    mooncake_bootstrap_port: 25201

# Decode stage (KV consumer)
kv_transfer_config:
  kv_connector: "MooncakeConnector"
  kv_role: "kv_consumer"
  kv_rank: 1
  kv_parallel_size: 2
  engine_id: "omni-thinker-decode"
  kv_connector_extra_config:
    mooncake_bootstrap_port: 25202
```

- `kv_role`: `"kv_producer"` for prefill, `"kv_consumer"` for decode.
- `kv_rank`: Must be unique across the P/D pair (0 for producer, 1 for consumer).
- `kv_parallel_size`: Total number of KV engines (always 2 for a single P/D pair).
- `engine_id`: Unique identifier for each engine. The orchestrator uses this to
  tell the decode engine where to pull KV from.
- `mooncake_bootstrap_port`: Must be different for each engine to avoid port
  conflicts.

### MooncakeStoreConnector Config (Multi-Node)

When using `qwen3_omni_moe_pd_multiconnector.yaml`, configure the Mooncake
backend:

```yaml
runtime:
  connectors:
    connector_of_mooncake:
      name: MooncakeStoreConnector
      extra:
        host: "127.0.0.1"                              # Local bind address
        metadata_server: "http://127.0.0.1:8181/metadata"  # Mooncake metadata server
        master: "127.0.0.1:50051"                       # Mooncake master address
        segment: 512000000                              # 512MB segment size
        localbuf: 64000000                              # 64MB local buffer
        proto: "tcp"                                    # Transport protocol
```

For multi-node deployments, replace `127.0.0.1` with the actual network addresses
of the Mooncake metadata server and master.

## Using Tensor Parallelism with PD

When the thinker model requires tensor parallelism (e.g., TP=2), both prefill
and decode stages must use the same TP size:

```yaml
# Example: TP=2 for both stages (requires 5 GPUs total)
stage_args:
  - stage_id: 0
    is_prefill_only: true
    runtime:
      devices: "0,1"          # 2 GPUs for prefill
    engine_args:
      tensor_parallel_size: 2
      kv_transfer_config:
        # ...

  - stage_id: 1
    is_decode_only: true
    runtime:
      devices: "2,3"          # 2 GPUs for decode
    engine_args:
      tensor_parallel_size: 2
      kv_transfer_config:
        # ...

  - stage_id: 2
    runtime:
      devices: "4"            # 1 GPU for talker + code2wav
    # ...
```

## Comparison: Standard vs PD Disaggregation

| Aspect | Standard (3-Stage) | PD Disaggregation (4-Stage) |
| :--- | :--- | :--- |
| Config | `qwen3_omni_moe.yaml` | `qwen3_omni_moe_pd_separation.yaml` |
| GPU count | 2+ GPUs | 3+ GPUs |
| Thinker stages | 1 (prefill + decode) | 2 (separate prefill and decode) |
| KV connector | Not needed | Required (MooncakeConnector) |
| Throughput | Lower (prefill blocks decode) | Higher (prefill and decode overlap) |
| Latency | Lower per-request | Slightly higher per-request (KV transfer overhead) |

## Troubleshooting

- **KV transfer timeout**: Ensure both prefill and decode engines can reach each
  other via the Mooncake bootstrap ports. Check firewall rules and network
  connectivity.
- **Port conflicts**: Each engine must use a unique `mooncake_bootstrap_port`.
  If running multiple PD pairs, assign non-overlapping port ranges.
- **TP size mismatch**: If prefill and decode use different `tensor_parallel_size`,
  KV cache block layouts will be incompatible. Always use the same TP size.
- **OOM errors**: The prefill stage processes the full prompt in one pass.
  Increase `gpu_memory_utilization` or reduce `max_num_batched_tokens` if OOM
  occurs during prefill.
