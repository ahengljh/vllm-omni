# PD Disaggregation PR 改动讲解文档

> 本文档详细讲解 PR 中每个文件的改动内容、设计动机和技术细节，用于 Code Review 和技术分享。

## 概览

本 PR 为 vllm-omni 的 thinker 阶段实现了 Prefill-Decode（PD）分离。将原本单一的 thinker stage 拆分为独立的 **prefill** 和 **decode** 两个 stage，通过 vLLM 原生的 KV connector（如 MooncakeConnector）进行 KV cache 传输，从而实现计算资源的更优分配。

### 整体架构变化

```
原始 3-stage pipeline:
  Stage 0: Thinker (prefill + decode) → Stage 1: Talker → Stage 2: Code2Wav

PD 分离后 4-stage pipeline:
  Stage 0: Thinker Prefill (kv_producer)  ─── KV Transfer ──→  Stage 1: Thinker Decode (kv_consumer)
                                                                       ↓
                                                                Stage 2: Talker → Stage 3: Code2Wav
```

### 改动统计

| 文件 | 新增行 | 类型 |
|------|--------|------|
| `vllm_omni/entrypoints/omni.py` | +560 | 核心编排逻辑 |
| `vllm_omni/entrypoints/async_omni.py` | +140 | 异步（在线服务）路径 |
| `vllm_omni/entrypoints/omni_llm.py` | +68 | KV flush 辅助 |
| `vllm_omni/entrypoints/omni_stage.py` | +111 | Stage worker 支持 |
| `vllm_omni/distributed/kv_transfer/mooncake_pd_adapter.py` | +275 | 新文件 |
| `vllm_omni/distributed/kv_transfer/connector_installer.py` | +105 | 新文件 |
| `vllm_omni/distributed/kv_transfer/__init__.py` | +13 | 新文件 |
| `vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py` | +65 | 模型层 padding 安全 |
| `vllm_omni/model_executor/stage_input_processors/qwen3_omni.py` | +154 | PD embedding 合并 |
| `vllm_omni/model_executor/stage_configs/qwen3_omni_moe_pd_separation.yaml` | +199 | 新 YAML 配置 |
| `tests/entrypoints/test_pd_disaggregation.py` | +1329 | 单元测试 |
| `tests/model_executor/.../test_qwen3_omni_stage_processors.py` | +1607 | stage processor 测试 |
| `tests/e2e/offline_inference/test_qwen3_omni_pd.py` | +66 | E2E 离线测试 |
| `tests/e2e/online_serving/test_qwen3_omni_pd.py` | +122 | E2E 在线测试 |
| `tests/e2e/stage_configs/qwen3_omni_pd_ci.yaml` | +184 | CI 测试配置 |

---

## 一、核心编排层：`vllm_omni/entrypoints/omni.py`

这是最核心的改动文件（+560 行）。在 `OmniBase` 类中新增了完整的 PD 分离检测、验证、准备和路由逻辑。

### 1.1 PD 分离检测 `_detect_pd_separation()`

```python
def _detect_pd_separation(self) -> tuple[int, int] | None:
```

**作用**：在 `__init__` 中扫描 `stage_list`，通过 YAML 配置中的 `is_prefill_only` 和 `is_decode_only` 标志自动检测 prefill-decode stage 对。

**工作原理**：
- 第一步：遍历所有 stage，收集带 `is_prefill_only=True` 的 stage 到 `prefill_by_id` 字典
- 第二步：遍历 decode stage，通过 `engine_input_source` 匹配其上游 prefill stage
- 返回 `(prefill_stage_id, decode_stage_id)` 元组
- 复杂度为 O(d×s)，其中 d=decode stage 数量（通常为 1），s=每个 decode stage 的 source 数量（通常 1~2）

### 1.2 配置验证 `_validate_pd_separation_config()`

```python
def _validate_pd_separation_config(self) -> None:
```

**验证项（6 项）**：
1. **kv_transfer_config 存在性**：两个 stage 都必须配置 `kv_transfer_config`
2. **kv_role 正确性**：prefill 必须是 `kv_producer` 或 `kv_both`；decode 必须是 `kv_consumer` 或 `kv_both`
3. **engine_input_source 匹配**：decode stage 的 source 必须包含 prefill stage
4. **kv_connector 一致性**：prefill 和 decode 必须使用相同的 connector
5. **kv_buffer 参数一致性**：buffer device 和 size 必须匹配
6. **tensor_parallel_size 一致性**（Review R2 新增）：MooncakeConnector 不支持异构 TP

### 1.3 Prefill 采样参数准备 `_prepare_prefill_sampling_params()`

```python
def _prepare_prefill_sampling_params(self, req_id: str, sp: SamplingParams) -> SamplingParams:
```

**核心逻辑**：
```python
sp = sp.clone()
sp.max_tokens = 1          # 只生成一个 token，目的是触发 KV cache 保存
sp.min_tokens = 1
sp.stop = []               # 清空 stop 条件 ← Review R2 新增
sp.stop_token_ids = []     # 清空 stop token IDs ← Review R2 新增
sp.include_stop_str_in_output = False  # ← Review R2 新增
```

**为什么要清空 stop 条件？（关键 bugfix）**

MooncakeConnector 的 `request_finished()` 方法内部检查 `finish_reason`：
- 只有 `FINISHED_LENGTH_CAPPED`（即 `max_tokens` 耗尽）才会正常进行 KV 传输
- 如果 finish_reason 是 `stop`（命中了 stop token/string），connector 会**取消** KV 传输

如果 prefill stage 的采样参数中包含 stop 条件，且碰巧第一个生成的 token 命中了 stop 条件，KV 传输就会被取消，decode stage 无法加载 KV cache，推理失败。清空所有 stop 条件确保 prefill 一定以 `length` 原因结束。

另外，注入 `kv_transfer_params` 关键字段：
- `do_remote_decode: True` — 告诉 prefill engine 的 connector 要进行远端 KV 发送
- `do_remote_prefill: False` — 这不是 decode 端
- `transfer_id` — 用于跟踪传输的唯一标识

### 1.4 PD 路由逻辑（同步路径）

在 `_run_generation()` 的主循环中，当输出从 prefill stage 到达后，路由到 decode stage：

**合并顺序（5步，与 async 路径一致）**：
1. 初始化 role flags（`do_remote_decode=False`, `do_remote_prefill=True`）
2. 合并用户提供的 `kv_transfer_params`
3. 合并 config 中的 connector 信息（`prefill_engine_id`, `prefill_bootstrap_addr`）
4. 合并 prefill 输出中携带的参数（`remote_request_id`, `remote_host`, `remote_port`）
5. 重新确认 role flags（确保上面的合并不会覆盖关键标志）

**关键设计**：PD 路由时使用**原始 prompt**（而非 prefill 输出）作为 decode 引擎输入，因为 decode 引擎需要重新编码 prompt 以匹配其内部 token 映射。

### 1.5 KV 参数存储与清理

```python
self._pd_kv_params_by_req: dict[str, dict[str, Any]] = {}
self._pd_kv_params_lock = threading.Lock()
```

**生命周期**：
- **存储**：prefill 输出到达时从 engine_outputs 提取 `kv_transfer_params` 存入
- **消费**：PD 路由时 `_pop_pd_kv_params()` 弹出并合并到 decode 参数
- **清理路径（3 条）**：
  - 错误路径：error 分支调用 `_drop_pd_kv_params(req_id)`
  - 完成路径：请求完全结束时调用
  - 防御性清理（Review R2 新增）：`_run_generation()` 结束后清理所有剩余条目

**线程模型**（Review R2 注释）：同步路径是单线程的——store 和 pop 在同一循环中顺序发生。Lock 主要是为异步路径 `AsyncOmni` 准备的。

### 1.6 采样参数自动复制

用户给出的采样参数是针对**逻辑 stage**（thinker, talker, code2wav）的，不需要知道 thinker 被拆分。当参数个数 = stage 数 - 1 时，自动复制 thinker 的参数给 decode stage。

### 1.7 其他改动

- **命名常量**（Review R2）：`_DEFAULT_MOONCAKE_BOOTSTRAP_PORT = 25201`，原来硬编码
- **`_to_dict()` 通用转换**：支持 dict/dataclass/pydantic/DictConfig 等类型
- **thin wrapper 注释**（Review R2）：解释 `_kv_cfg_to_dict` 和 `_normalize_kv_transfer_params` 为何是独立方法
- **错误消息改进**（Review R2）：`_validate_pd_separation_config` 中包含 `type(cfg).__name__`
- **日志级别**（Review R2）：`_prepare_prefill_sampling_params` 从 `logger.info` → `logger.debug`

---

## 二、异步路径：`vllm_omni/entrypoints/async_omni.py`

`AsyncOmni` 继承 `OmniBase`，用于 `vllm serve` 在线推理场景（+140 行）。

### 2.1 核心改动

与同步路径对称，但有以下差异：

| 差异点 | 同步路径 (Omni) | 异步路径 (AsyncOmni) |
|--------|----------------|---------------------|
| SP 自动复制日志级别 | DEBUG | WARNING（并告知如何消除） |
| PD 路由日志级别 | INFO | DEBUG（每请求触发，避免过多日志） |
| KV params 存储 | `_pd_kv_params_by_req` | 直接从 engine_outputs 提取 |
| prefill 输出追踪 | 无 | 额外 DEBUG 日志（token 数、kv_params） |

### 2.2 remote_request_id 缺失警告

如果 decode_kv_params 中缺少 `remote_request_id`，输出详细 WARNING：
- decode 引擎的 MooncakeConnector 会使用**自己**的 internal request_id
- 这与 prefill 引擎的 request_id 不匹配 → KV 传输**将会失败**
- 解决方案：应用 monkey_patch 补丁

---

## 三、KV 传输层（3 个新文件）

### 3.1 `mooncake_pd_adapter.py`（275 行）

**问题**：vLLM 的 MooncakeConnector 假设 prefill 和 decode 引擎使用相同 request_id 索引 KV cache。但在 vllm-omni 中两个引擎是独立 stage worker，各自分配不同 request_id。

**解决方案**：创建 `PatchedMooncakeConnector` 子类，修正 request_id 映射。

#### 4 个关键方法重写：

| 方法 | 侧 | 作用 |
|------|-----|------|
| `request_finished()` | prefill | 将 prefill 的 internal request_id 注入为 `remote_request_id` |
| `add_new_req()` | decode | 先调 super()，再用 `PatchedRecvReqMeta` 携带 remote_request_id |
| `group_kv_pull()` | decode | save-patch-restore 模式：临时将 _reqs_need_recv key 替换为 remote_id |
| `receive_kv()` | decode | 清理已完成的 remote→local 映射 |

**`group_kv_pull()` 的 save-patch-restore 模式**（Review R2 注释增强）：
```
1. 保存原始 _reqs_need_recv
2. 将 key 从 local_request_id 替换为 remote_request_id
3. 调用 super().group_kv_pull()（基类直接读 self._reqs_need_recv）
4. 将未消费的条目恢复回原始 local key
```
不能用 copy-and-return 方式，因为 `super()` 直接读取 `self._reqs_need_recv`。

**`__qualname__` 设置**（Review R2 修复）：
```python
PatchedMooncakeConnector.__qualname__ = _OriginalMooncakeConnector.__qualname__
```
使用动态引用（而非硬编码字符串），确保 vLLM 内部 isinstance 检查正常。

### 3.2 `connector_installer.py`（105 行）

在 stage worker 启动时替换 vLLM 模块中的 MooncakeConnector 引用。

**4 步流程**：
```
Step 0: 版本兼容性检查（vLLM >= 0.8.0，warn 不 block）← Review R2 新增
Step 1: import 原始 MooncakeConnector
Step 2: 创建 patched 子类
Step 3: 替换定义模块中的引用
Step 4: 遍历 sys.modules 替换所有已导入模块中的引用
```

使用 `_patched` 全局标志确保幂等。

---

## 四、Stage Worker 支持

### 4.1 `omni_stage.py`（+111 行）

#### kv_transfer_params 备份机制

```python
if "_kv_transfer_params" not in payload:
    sp = payload.get("sampling_params")
    if sp is not None and hasattr(sp, "extra_args") and sp.extra_args:
        kv_tp = sp.extra_args.get("kv_transfer_params")
        if kv_tp is not None:
            payload["_kv_transfer_params"] = dict(kv_tp)
```

**背景**：SamplingParams 是 `msgspec.Struct`，使用 `omit_defaults=True`。在 pickle 跨进程通信中 `extra_args` 可能被丢弃。备份到 payload 顶层 key 确保可靠传输。

**注释更新**（Review R2）：添加 `# TODO: open vLLM issue if confirmed reproducible`。

#### Stage Worker 中的 Monkey Patch

在 `_stage_worker()` 和 `_stage_worker_async()` 中，engine 构建前调用 `apply_mooncake_connector_patch()`。

#### finish_reason 检查

prefill stage 完成后检查 finish_reason，如果不是 `'length'`（即 `FINISHED_LENGTH_CAPPED`），输出 WARNING 提示 KV 传输可能被跳过。

### 4.2 `omni_llm.py`（+68 行）

#### `_flush_kv_connector_sends()`

**解决的时序问题**：

vLLM EngineCore 的 `step()` 执行顺序：
```
1. schedule() → 调用 build_connector_meta() 清空 _reqs_need_send
2. execute_model() → 模型前向传播
3. update_from_output() → 调用 request_finished() 添加新条目到 _reqs_need_send
```

在 batch 模式下，如果 `request_finished()` 在最后一步添加了条目，而引擎循环已结束，这些条目永远不会被发送到 worker。

**解决方案**：在 `_run_engine()` 返回前检查 pending 的 `_reqs_need_send`，创建空的 `SchedulerOutput` 附上 connector metadata，执行一次空 model step。

**注意**：这个方法深入 vLLM 内部实现，没有公开的"flush" API。经测试兼容 vLLM >= 0.8.x。

---

## 五、模型层改动

### 5.1 `qwen3_omni.py`（模型层）—— Zero-Padding 安全阈值

```python
_PD_PAD_THRESHOLD = 512

if pad_len > _PD_PAD_THRESHOLD:
    logger.warning(
        "[PD] Unexpectedly large embed padding: %d tokens (threshold=%d). "
        "This may indicate a bug in PD disaggregation.",
        pad_len, _PD_PAD_THRESHOLD,
    )
```

**背景**：PD 分离时 talker 需要拼接 prefill 和 decode 的 embedding/hidden state。实际 token 数可能与 `result_ids` 长度不完全匹配，需要 zero-padding。

**Review R2 改动**：增加 512 token 阈值检查。超过阈值输出 WARNING（不是 exception，避免误报中断推理）。对 `thinker_embed` 和 `thinker_hidden` 分别检查。

### 5.2 `stage_input_processors/qwen3_omni.py` —— PD Embedding 合并

#### 新增 `_merge_pd_embeddings()` 函数

```python
def _merge_pd_embeddings(
    decode_embed, decode_hidden, prefill_mm,
    device, expected_total=None
) -> tuple[Tensor, Tensor]:
```

**核心逻辑**：
1. 从 prefill multi-modal 数据中提取 embed（key="0"）和 hidden（key="24"）
2. 计算 overlap：`overlap = prefill_len + decode_len - expected_total`
3. 跳过 decode embedding 中的 overlap 部分
4. 拼接：`[prefill_embed | decode_embed[overlap:]]`

**为什么有 overlap？** decode 引擎收到的是完整 prompt（不是从 prefill 位置继续），它会重新做 prompt 编码。因此 decode 的 embedding 前 N 个 token 与 prefill 的最后 N 个 token 重叠。

#### 常量定义

```python
_EMBED_LAYER_KEY = "0"      # 对应 thinker 模型第 0 层输出（embedding）
_HIDDEN_LAYER_KEY = "24"    # 对应 thinker 模型第 24 层输出（hidden state）
```

来自 `TalkerConfig.accept_hidden_layer` 配置。如果模型结构变化需同步更新。

#### 日志级别（Review R2）

`_merge_pd_embeddings` 和 `thinker2talker` 中的 `logger.info` → `logger.debug`。

---

## 六、Stage 配置

### 6.1 `qwen3_omni_moe_pd_separation.yaml`（199 行，新文件）

完整的 4-stage PD 分离配置模板，包含详细注释说明：

```yaml
stage_args:
  - stage_id: 0
    is_prefill_only: true           # PD 检测标志
    engine_args:
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_producer"      # 发送 KV cache
        engine_id: "omni-thinker-prefill"  # 必须显式设置

  - stage_id: 1
    is_decode_only: true
    engine_input_source: [0]        # 从 stage 0 接收
    engine_args:
      kv_transfer_config:
        kv_connector: "MooncakeConnector"
        kv_role: "kv_consumer"      # 接收 KV cache
        engine_id: "omni-thinker-decode"

  - stage_id: 2   # Talker
  - stage_id: 3   # Code2Wav
```

**关键约束**：
- 两个 stage 的 `tensor_parallel_size` 必须相同
- 两个 stage 的 `mooncake_bootstrap_port` 必须不同
- `kv_parallel_size` 必须设为 2（表示有 2 个参与 KV 传输的引擎）

---

## 七、测试

### 7.1 单元测试：`test_pd_disaggregation.py`（1329 行）

使用 mock/fake 对象避免 GPU 依赖，覆盖 OmniBase 中所有 PD 相关方法。

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|----------|
| TestDetectPDSeparation | 4 | PD 对检测逻辑 |
| TestValidatePDConfig | 6 | role/connector/TP 验证 |
| TestGetPDConnectorInfo | 3 | connector info 提取 |
| TestPreparePrefillSamplingParams | 4 | prefill SP 准备 |
| TestSamplingParamsAutoDuplication | 1 | SP 自动复制 |
| TestNormalizeKVTransferParams | 3 | 参数归一化 |
| TestKvCfgToDict | 3 | 配置转 dict |
| TestPDRouting | 3 | PD 路由逻辑 |
| TestKVParamsCleanup | 4 | KV 参数清理 |
| TestPDYAMLConfig | 1 | YAML 配置解析 |
| TestMooncakeConnectorPatch | 4 | monkey patch |
| **TestPrefillStopNeutralization** | **4** | **stop 条件清空** (R2) |
| **TestPDFailureModes** | **3** | **错误路径/内存泄漏** (R2) |
| **TestTPSizeValidation** | **3** | **TP 大小验证** (R2) |

### 7.2 Stage Processor 测试：`test_qwen3_omni_stage_processors.py`（1607 行）

| 测试类 | 测试内容 |
|--------|----------|
| TestMergePDEmbeddings | PD embedding 合并（overlap/no-overlap/empty） |
| TestGetPrefillStage | prefill stage 查找 |
| TestThinker2TalkerNonPD | 非 PD 模式 thinker→talker |
| TestThinker2TalkerPDMode | PD 模式 thinker→talker |
| TestTalker2Code2Wav | talker→code2wav |
| TestPDAudioPipelineIntegration | PD 音频全流程集成 |

### 7.3 E2E 测试

#### 离线测试 `tests/e2e/offline_inference/test_qwen3_omni_pd.py`

```python
@hardware_test(res={"cuda": "H100"}, num_cards=3)
def test_pd_text_only(omni_runner, omni_runner_handler):
    """纯文本输入 → 文本输出"""

def test_pd_video_to_audio(omni_runner, omni_runner_handler):
    """视频输入 → 音频输出（完整 4-stage 管道）"""
```

#### 在线测试 `tests/e2e/online_serving/test_qwen3_omni_pd.py`

```python
@hardware_test(res={"cuda": "H100"}, num_cards=3)
def test_pd_text_to_text(omni_server, openai_client):
    """通过 OpenAI API 测试文本→文本"""

def test_pd_mix_to_text_audio(omni_server, openai_client):
    """通过 OpenAI API 测试文本→文本+音频"""
```

---

## 八、Review Round 2 改动索引

以下改动专门针对 PR #1303 review comments：

| # | 改动 | 审阅者 | 文件 |
|---|------|--------|------|
| 1 | 清空 stop/stop_token_ids 条件 | codex-connector P1 | omni.py |
| 2 | 硬编码端口 25201 → 命名常量 | hsliuustc0106 | omni.py |
| 3 | tensor_parallel_size 验证 | hsliuustc0106 | omni.py |
| 4 | 错误消息包含 type(cfg).__name__ | hsliuustc0106 + lishunyang12 | omni.py |
| 5 | O(d×s) 复杂度注释 | lishunyang12 | omni.py |
| 6 | thin wrapper 说明注释 | lishunyang12 | omni.py |
| 7 | 线程模型注释 | hsliuustc0106 | omni.py |
| 8 | logger.info → logger.debug | lishunyang12 | omni.py, async_omni.py, qwen3_omni.py |
| 9 | defense-in-depth cleanup | hsliuustc0106 | omni.py |
| 10 | SP 自动复制升级为 WARNING | hsliuustc0106 | async_omni.py |
| 11 | 合并顺序注释 | lishunyang12 + codex-connector | async_omni.py |
| 12 | vLLM 版本兼容性检查 | hsliuustc0106 | connector_installer.py |
| 13 | __qualname__ 动态引用 | lishunyang12 | mooncake_pd_adapter.py |
| 14 | save-patch-restore 注释 | lishunyang12 | mooncake_pd_adapter.py |
| 15 | padding 阈值警告 | hsliuustc0106 + lishunyang12 | qwen3_omni.py (model) |
| 16 | backup 机制注释 + TODO | lishunyang12 | omni_stage.py |
| 17 | stop 清空测试 | Review R2 | test_pd_disaggregation.py |
| 18 | failure mode 测试 | hsliuustc0106 | test_pd_disaggregation.py |
| 19 | TP 验证测试 | Review R2 | test_pd_disaggregation.py |
