# Bagel Benchmark And Validation Plan

## Scope

This plan covers Bagel functional validation, `ar->dit` split equivalence,
exploratory single-GPU performance probing, and matched-budget performance
comparison.

## Environment Guardrails

- Work only in the container whose `hostname` is exactly `Bagel-perf`
- Work only in repo path `/home/fq9hpsacuser01/vllm-omni`
- Work only on branch `main` unless the user explicitly changes the rule
- Do not treat the default two-stage
  `vllm_omni/model_executor/stage_configs/bagel.yaml` path as the baseline
- Treat `/scripts` as local helper space, not as official tracked benchmark code

## Agent Coordination

Recommended Bagel sub-agent template:

- `Bagel-Ops`: remote execution only; owns process cleanup, service start, request sending, benchmark execution, and artifact collection
- `Bagel-Validation`: result validation only; owns output correctness checks, fairness checks, and acceptance decisions
- `Bagel-Analysis`: code and benchmark analysis only; owns code-path review, benchmark-framework review, and issue localization
- main agent remains the coordinator; owns plan updates, task routing, and final conclusions

Rules:

- before spawning any new sub-agents, first inspect whether the current session already has active Bagel-role sub-agents
- check for role matches by responsibility first, not only by nickname
- if a matching agent already exists, reuse it and resync it to the latest remote plan instead of creating a duplicate
- if some roles are missing, create only the missing roles
- if creating fresh agents, prefer creating them in this order: `Bagel-Ops`, `Bagel-Validation`, `Bagel-Analysis`
- keep at most one active Bagel sub-agent per role unless the user explicitly asks for additional parallel agents
- when the Bagel plan changes, broadcast the updated context to all active Bagel sub-agents

## Baseline Definition

The baseline is:

- exactly one stage
- `stage_type: diffusion`
- one `DiT` stage with full weights
- that single `DiT` stage must support both `text->image` and `image->image`

The baseline is not:

- the default two-stage `AR + DiT` Bagel pipeline
- a generic "merged AR/DiT" description

## Test Contract Freeze

Before any comparison, record and freeze:

- commit id
- container hostname
- repo path
- model path / weight version
- API path used for requests
- prompt set and input-image set
- seed
- resolution
- num_inference_steps
- guidance / negative prompt / other sampling params
- batch size and concurrency model
- stage config
- GPU allocation and parallel settings
- warmup policy and measurement window

Do not compare runs with mismatched contracts.

## Official Benchmark Decision

Official tracked benchmark code for diffusion performance lives under:

- `benchmarks/diffusion/README.md`
- `benchmarks/diffusion/diffusion_benchmark_serving.py`
- `benchmarks/diffusion/backends.py`
- `vllm_omni/entrypoints/cli/benchmark/serve.py`

Rules:

- use the official tracked diffusion benchmark as the primary performance path
- treat `/scripts/bagel/...` and `/scripts/benchmark/...` as local helper or plan
  space only
- do not present `/scripts/bagel/vllm_omni/*` as official benchmark coverage
- PR `#1805` is reference material, not a drop-in Bagel solution
- only selectively absorb ideas from PR `#1805`, such as extra request fields,
  if the current official benchmark is missing a capability needed for Bagel

## Current Official vLLM-Omni DiT Performance Path

Current tracked diffusion benchmark behavior:

- main entrypoint: `benchmarks/diffusion/diffusion_benchmark_serving.py`
- backend `vllm-omni`: uses `/v1/chat/completions`
- backend `openai`: uses `/v1/images/generations`
- tasks supported in the benchmark: `t2i`, `i2i`, `t2v`, `i2v`

Current tracked dataset modes:

- `vbench`
- `trace`
- `random`

Dataset details relevant to Bagel:

- for `t2i`, `vbench` reuses VBench text prompts
- for `i2i`, `vbench` loads the VBench i2v dataset and its image paths
- for `trace`, the benchmark can use per-request fields such as `width`,
  `height`, `num_inference_steps`, `seed`, and optional `slo_ms`
- for `random`, the benchmark generates synthetic prompts for smoke tests

Current limitation on this branch:

- the current `main` branch does not yet contain PR `#1805` additions for
  `negative_prompt`, `guidance_scale`, or `cfg_scale` in the official diffusion
  benchmark CLI
- if Bagel benchmarking needs those fields, selectively port the minimal
  required changes instead of adopting the whole PR blindly

## Local Custom Benchmark Status

Local helper scripts currently present under `/scripts` are custom and ignored by
Git. They can be used for quick experiments, but they are not the official
benchmark source of truth.

Current custom coverage status:

- `scripts/bagel/vllm_omni/run_benchmark.sh`
- `scripts/bagel/vllm_omni/benchmark_bagel_online.py`

Current custom behavior:

- measures only `text->image`
- targets `/v1/images/generations`
- currently hardcodes a single clean prompt in Python
- does not provide tracked `image->image` coverage

Use these scripts only as temporary helpers when the official benchmark path is
not yet sufficient for a Bagel-specific question.

## API Rule For Formal Validation

For formal Bagel correctness and performance runs:

- prefer `/v1/images/generations` for `text->image`
- prefer `/v1/images/edits` for `image->image`
- do not rely on `/v1/chat/completions` as the final Bagel benchmark path until
  request parameters are verified to take effect in the current setup

Reason:

- current server logs show that the `/v1/chat/completions` path may ignore
  `width`, `height`, `num_inference_steps`, `negative_prompt`, and related
  image-generation request fields in the current single-stage Bagel setup

## Phase 1: Single-Stage Baseline Correctness

Goal:

- verify that the single-stage full-weight `DiT` baseline runs correctly

Required modalities:

1. `text->image`
2. `image->image`

Requirements:

- use a fixed golden set for each modality
- keep input prompt, input image, seed, and generation params fixed
- archive outputs and logs
- perform manual semantic review before accepting the baseline

Suggested acceptance examples:

- `text->image`: prompt semantics are clearly reflected in the output image
- `image->image`: output preserves the main subject and scene intent while
  applying the requested edit or style change

Current bootstrap note:

- a temporary one-stage diffusion config has already been validated as runnable
  in `/tmp/bagel_single_stage.yaml`
- if a repo-owned baseline config is needed, add it under
  `scripts/benchmark/bagel/` after this plan is approved

## Phase 2: `ar->dit` Split Equivalence

Goal:

- verify that the split `AR -> DiT` path produces outputs consistent with the
  accepted single-stage baseline

Requirements:

- use exactly the same golden set as the baseline
- use the same seeds and generation parameters
- record the exact split stage config and GPU allocation

Checks:

1. if deterministic AR intermediate outputs are available, compare them first
2. compare final images against the single-stage baseline
3. document any allowed tolerance before running the test

Do not call the split path valid if either modality fails equivalence.

## Phase 3: Single-GPU Performance Probe

Goal:

- probe Bagel performance on the smallest setup before any formal multi-GPU
  comparison or 8-GPU tuning

Probe matrix:

1. one-stage full-weight `DiT`, single GPU, `text->image`
2. one-stage full-weight `DiT`, single GPU, `image->image`
3. split `AR -> DiT`, single GPU, `text->image`
4. split `AR -> DiT`, single GPU, `image->image`

Rules:

- use the official tracked diffusion benchmark first wherever it can cover the
  target task correctly
- if the official benchmark path is insufficient for Bagel on current `main`,
  document the exact gap before using a local helper script or a minimal patch
- for the single-GPU split probe, pin both `AR` and `DiT` to the same GPU when
  the path is runnable, so the result exposes split orchestration overhead on a
  minimal setup
- if a split path cannot run on a single GPU, record the exact failure and the
  smallest runnable fallback setup
- use a small fixed probe set derived from the accepted baseline golden set
- do not claim the final performance winner from this phase; this phase is only
  to establish a bottom-line reference and uncover obvious bottlenecks

Measure at least:

- end-to-end latency
- success rate
- output artifact count
- basic GPU memory footprint if available

## Phase 4: Performance Comparison

Goal:

- compare the split path against matched single-stage baselines under the same
  total GPU budget

Prerequisite:

- complete the single-GPU probe first, then decide whether any config or API
  path needs correction before spending time on larger-scale runs

Measure at least:

- end-to-end latency
- stage latency breakdown
- throughput
- GPU utilization
- peak GPU memory
- failure / timeout rate

Rules:

- separate latency conclusions from throughput conclusions
- report `p50`, `p95`, and `p99` when sample count is sufficient
- use the same workload mix, same warmup policy, and same measurement window
- archive configs, logs, and summary metrics for every run

Comparison sets:

1. split `AR -> DiT` vs one single-stage full-weight `DiT` baseline under the
   same total GPU budget
2. split `AR -> DiT` vs replicated single-stage baselines under the same total
   GPU budget when the question is throughput or cluster efficiency

## Phase 5: Split Tuning

Goal:

- tune `AR -> DiT` resource allocation and internal parallel strategy

Recommended order:

1. fix the total GPU budget
2. sweep `AR:DiT` stage split, for example `1:7`, `2:6`, `3:5`, `4:4`
3. hold the best stage split fixed
4. sweep `DiT` internal parallel strategy and degree
5. compare the best split setup against the required single-stage baseline set

Rules:

- change one variable family at a time
- do not mix stage-budget conclusions with `DiT` parallel-mode conclusions
- for an 8-GPU case, compare the best split result against an 8-GPU
  single-stage baseline

## Completion Gates

Bagel validation is complete only when all requested gates are closed with
fresh artifacts:

1. single-stage baseline correctness is accepted for `text->image` and
   `image->image`
2. `ar->dit` split equivalence is accepted against the same baseline set
3. single-GPU probe is archived and reviewed
4. matched-baseline performance comparison is complete with archived metrics and
   artifacts

The following are not completion states:

- service started
- health is green
- one modality passed
- output looks similar
- numbers improved in one ad hoc run

## Artifacts To Keep

For each accepted run, keep:

- stage config
- serve command
- request command
- logs
- output images
- summary metrics
- manual review conclusion

## Next Execution Order

1. materialize a repo-owned single-stage baseline config under
   `scripts/benchmark/bagel/` if needed
2. switch formal `text->image` and `image->image` validation to image-specific
   APIs
3. use the official diffusion benchmark as the default starting point for Bagel
   performance work
4. selectively port only the missing Bagel-relevant pieces from PR `#1805` if
   the official benchmark needs extra request fields
5. freeze the first baseline golden set and acceptance record
6. run the single-GPU one-stage `DiT` probe for `text->image` and `image->image`
7. run the single-GPU split `AR -> DiT` probe for `text->image` and
   `image->image`
8. start matched-budget performance benchmarking
9. start larger-scale split tuning only after the earlier phases are stable
