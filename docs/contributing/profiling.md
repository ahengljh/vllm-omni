# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni supports two profiling approaches:
- **Nsight Systems (nsys)** — GPU-level tracing during online serving via HTTP endpoints
- **PyTorch Profiler** — detailed CPU/CUDA traces for offline inference scripts

### 1. Nsight Systems Profiling for Online Serving

NVIDIA Nsight Systems (`nsys`) can capture GPU-level traces while the server is running. The API server exposes `/start_profile` and `/stop_profile` HTTP endpoints that signal nsys via `torch.cuda.profiler.start()` / `stop()`.

**Step 1 — Launch the server under nsys:**

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

`--capture-range=cudaProfilerApi` tells nsys to sit idle until `torch.cuda.profiler.start()` is called in a worker process. `--capture-range-end=repeat` allows multiple start/stop cycles in the same session.

**Step 2 — Start profiling:**

```bash
curl -X POST http://localhost:8091/start_profile
```

**Step 3 — Send requests:**

```bash
curl -X POST http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Omni-7B","messages":[{"role":"user","content":"Hello"}]}'
```

**Step 4 — Stop profiling:**

```bash
curl -X POST http://localhost:8091/stop_profile
```

**Step 5 — Shut down the server** (Ctrl+C). nsys writes a `.nsys-rep` file in the current directory.

```bash
ls *.nsys-rep
nsys stats report1.nsys-rep
```

Open the `.nsys-rep` file in the Nsight Systems GUI for a detailed timeline of CUDA kernels, memory operations, and NVTX ranges.

### 2. Set the Output Directory (PyTorch Profiler)
Before running any offline profiling script, set this environment variable. The system detects this and automatically saves traces here.

```bash
export VLLM_TORCH_PROFILER_DIR=./profiles
```

### 3. Profiling Omni-Modality Models (Offline)

It is best to limit profiling to one iteration to keep trace files manageable.

```bash
export VLLM_PROFILER_MAX_ITERS=1
```

**Selective Stage Profiling**
The profiler is default to function across all stages. But It is highly recommended to profile specific stages by passing the stages list, preventing from producing too large trace files:
```python
# Profile all stages
omni_llm.start_profile()

# Only profile Stage 1
omni_llm.start_profile(stages=[1])
```

```python
# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for qwen omni
omni_llm.start_profile(stages=[0, 2])
```

**Python Usage**: Wrap your generation logic with `start_profile()` and `stop_profile()`.

```python
from vllm_omni import omni_llm

profiler_enabled = bool(os.getenv("VLLM_TORCH_PROFILER_DIR"))

# 1. Start profiling if enabled
if profiler_enabled:
    omni_llm.start_profile(stages=[0])

# Initialize generator
omni_generator = omni_llm.generate(prompts, sampling_params_list, py_generator=args.py_generator)

total_requests = len(prompts)
processed_count = 0

# Main Processing Loop
for stage_outputs in omni_generator:

    # ... [Output processing logic for text/audio would go here] ...

    # Update count to track when to stop profiling
    processed_count += len(stage_outputs.request_output)

    # 2. Check if all requests are done to stop the profiler safely
    if profiler_enabled and processed_count >= total_requests:
        print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")

        # Stop the profiler while workers are still active
        omni_llm.stop_profile()

        # Wait for traces to flush to disk
        print("[Info] Waiting 30s for workers to write trace files to disk...")
        time.sleep(30)
        print("[Info] Trace export wait time finished.")

omni_llm.close()
```


**Examples**:

1. **Qwen2.5-Omni**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)

2. **Qwen3-Omni**:   [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)


### 4. Profiling Diffusion Models (Offline)

Diffusion profiling is End-to-End, capturing encoding, denoising loops, and decoding.

**CLI Usage:**
```python

python image_to_video.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image qwen-bear.png \
    --prompt "A cat playing with yarn, smooth motion" \
    \
    # Minimize Spatial Dimensions (Optional but helpful):
    #    Drastically reduces memory usage so the profiler doesn't
    #    crash due to overhead, though for accurate performance
    #    tuning you often want target resolutions.
    --height 48 \
    --width 64 \
    \
    # Minimize Temporal Dimension (Frames):
    #    Video models process 3D tensors (Time, Height, Width).
    #    Reducing frames to the absolute minimum (2) keeps the
    #    tensor size small, ensuring the trace file doesn't become
    #    multi-gigabytes in size.
    --num_frames 2 \
    \
    # Minimize Iteration Loop (Steps):
    #    This is the most critical setting for profiling.
    #    Diffusion models run the same loop X times.
    #    Profiling 2 steps gives you the exact same performance
    #    data as 50 steps, but saves minutes of runtime and
    #    prevents the trace viewer from freezing.
    --num_inference_steps 2 \
    \
    --guidance_scale 5.0 \
    --guidance_scale_high 6.0 \
    --boundary_ratio 0.875 \
    --flow_shift 12.0 \
    --fps 16 \
    --output i2v_output.mp4

```

**Examples**:

1. **Qwen image edit**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)

2. **Wan-AI/Wan2.2-I2V-A14B-Diffusers**:   [https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)

### 5. Analyzing Omni Traces

Output files are saved to your configured ```VLLM_TORCH_PROFILER_DIR```.

**Output**
**Chrome Trace** (```.json.gz```): Visual timeline of kernels and stages. Open in Perfetto UI.

**Viewing Tools:**

- [Perfetto](https://ui.perfetto.dev/)(recommended)
- ```chrome://tracing```(Chrome only)

**Note**: vLLM-Omni reuses the PyTorch Profiler infrastructure from vLLM. See the official vLLM profiler documentation:  [vLLM Profiling Guide](https://docs.vllm.ai/en/stable/contributing/profiling/)
