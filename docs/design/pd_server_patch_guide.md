# Server-Side vLLM Patch Guide for PD Disaggregation

## Problem: Random/Garbled Output

When using PD disaggregation (Prefill-Decode separation), the decode engine may produce
random/garbled output instead of coherent text. This happens because **vLLM's
`InputProcessor.assign_request_id()` appends 8 random characters** to each request_id
internally:

```python
# vllm/v1/engine/input_processor.py
request.request_id = f"{request.external_req_id}-{random_uuid():.8}"
```

The **same** external request (e.g., `chatcmpl-xxx`) gets different internal IDs on prefill
(`chatcmpl-xxx-a1b1e5fe`) and decode (`chatcmpl-xxx-897efdb3`) engines. The upstream
`MooncakeConnector` uses these internal IDs as keys in `_reqs_need_send`, so the decode
worker's ZMQ request can never match the prefill worker's stored entry.

**Result**: "Request not found in reqs_need_send" → KV never sent → decode reads
uninitialized KV cache → garbled output.

## Solution: Apply the `remote_request_id` Patch

The fix introduces a `remote_request_id` field that carries the **prefill engine's internal
request_id** through the orchestrator to the decode engine, so the decode worker uses the
correct ID when requesting KV data from the prefill worker.

### Data Flow (after patch)

```
Prefill Engine                  Orchestrator              Decode Engine
─────────────                  ────────────              ────────────
request_finished()
  → remote_request_id =
    request.request_id ──────→ kv_transfer_params
    (prefiller's internal ID)   propagation ───────────→ kv_transfer_params
                                                          .remote_request_id
                                                        → group_kv_pull() uses
                                                          remote_request_id for
                                                          ZMQ request
                                                        → prefill worker matches
                                                          correctly
```

## Patch Source

Reference patch: https://github.com/natureofnature/vllm_014_pd_patch

The critical file is:
- `vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py`

## Apply the Patch

### Option 1: Direct file replacement (recommended)

Copy the patched `mooncake_connector.py` over the upstream version in your vLLM
installation:

```bash
# Find your vLLM installation path
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM installed at: $VLLM_PATH"

# Backup the original
CONNECTOR_PATH="$VLLM_PATH/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py"
cp "$CONNECTOR_PATH" "${CONNECTOR_PATH}.bak"

# Clone the patch repo and copy
git clone https://github.com/natureofnature/vllm_014_pd_patch /tmp/vllm_014_pd_patch
cp /tmp/vllm_014_pd_patch/vllm/distributed/kv_transfer/kv_connector/v1/mooncake_connector.py \
   "$CONNECTOR_PATH"

echo "Patch applied. Restart your vLLM servers."
```

### Option 2: Editable install patch

If you have vLLM installed in editable mode (`pip install -e .`):

```bash
cd /path/to/your/vllm
git remote add pd-patch https://github.com/natureofnature/vllm_014_pd_patch
git fetch pd-patch
# Cherry-pick or manually apply the mooncake_connector.py changes
```

## Key Changes in the Patch

### 1. `RecvReqMeta` — new `remote_request_id` field

```python
@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int
    remote_request_id: str  # Prefiller's internal request_id for KV lookup
```

### 2. `request_finished()` — returns `remote_request_id`

```python
def request_finished(self, request, block_ids):
    # ...
    return delay_free_blocks, dict(
        do_remote_prefill=True,
        do_remote_decode=False,
        remote_host=self.side_channel_host,
        remote_port=self.side_channel_port,
        remote_request_id=request.request_id,  # <-- NEW: prefiller's internal ID
    )
```

### 3. `_reqs_need_send` format change

```python
# Upstream:  dict[ReqId, tuple[Request, list[int]]]
# Patched:   dict[ReqId, list[int]]
self._reqs_need_send[request.request_id] = block_ids
```

### 4. `add_new_req()` — extracts `remote_request_id`

```python
def add_new_req(self, request_id, local_block_ids, kv_transfer_params, load_remote_cache=True):
    if load_remote_cache:
        self.reqs_to_recv[request_id] = RecvReqMeta(
            local_block_ids=local_block_ids,
            remote_host=kv_transfer_params["remote_host"],
            remote_port=kv_transfer_params["remote_port"],
            remote_request_id=kv_transfer_params.get("remote_request_id", request_id),
        )
```

### 5. `group_kv_pull()` — uses `remote_request_id` for ZMQ request

```python
def group_kv_pull(self, metadata):
    kv_pulls = defaultdict(list)
    for req_id, meta in metadata.reqs_to_recv.items():
        path = make_zmq_path("tcp", meta.remote_host, meta.remote_port + self.tp_rank)
        self.remote_to_local_req[meta.remote_request_id] = req_id  # Map back
        kv_pulls[path].append((meta.remote_request_id, meta.local_block_ids))
    return kv_pulls
```

### 6. `receive_kv()` — maps back from remote to local request_id

```python
async def receive_kv(self, path, req_blocks):
    # ... after ZMQ transfer completes ...
    for remote_id in req_ids:
        local_id = self.remote_to_local_req.pop(remote_id, None)
        if local_id is None:
            local_req_ids.append(remote_id)  # fallback
        else:
            local_req_ids.append(local_id)
    self.finished_recving_reqs.update(local_req_ids)
```

## Verification

After applying the patch, check the logs for:

1. **Prefill side**: `remote_request_id` should appear in `kv_transfer_params` output:
   ```
   [PD] Engine output: kv_transfer_params={'do_remote_prefill': True,
     'remote_host': '...', 'remote_port': ..., 'remote_request_id': 'chatcmpl-xxx-SUFFIX'}
   ```

2. **Orchestrator**: PD routing should show `remote_request_id` set:
   ```
   PD routing: stage-0→stage-1, remote_request_id=chatcmpl-xxx-SUFFIX
   ```

3. **Decode side**: No more "Request not found in reqs_need_send" errors.

## Important Notes

- The patch must be applied to **BOTH** prefill and decode vLLM server instances
- The vllm-omni orchestrator automatically propagates `remote_request_id` through
  `kv_transfer_params` — no orchestrator changes needed for this fix
- The patched `mooncake_connector.py` removes the bootstrap-server protocol and uses
  direct ZMQ connections (matching the vllm-omni PD disaggregation design)
