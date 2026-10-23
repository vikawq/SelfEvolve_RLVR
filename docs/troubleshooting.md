# Troubleshooting / 与上游文档的差异记录

协议要求：问题解决后按「现象 / 原因 / 解法」记录。任务包里的参数名、路径、版本只是指导；与已安装版本不一致时以源码和官方文档为准，并记录在此。

---

## 1. 环境类

### 1.1 根分区只剩 18G
- **现象**：`df -h /` 显示 183G 用了 91%，`/home` 有 857G 空闲。
- **原因**：系统盘容量小，conda env / pip 缓存 / HF 缓存默认会写到 `/`（`~/.cache` 实际在 `/home`，但 `/tmp` 在 `/`）。
- **解法**：conda 环境装在 `/home/angus/miniconda3`；所有脚本 export `TMPDIR=$HOME/tmp`、`PIP_CACHE_DIR=$HOME/.cache/pip`、`HF_HOME=$HOME/.cache/huggingface`。**不要**用 `sudo apt` 往 `/` 装大包。

### 1.2 直连 huggingface.co 超时
- **现象**：`curl https://huggingface.co/...` 12s 超时（HTTP 000）；`hf-mirror.com` 返回 200。
- **原因**：本机网络环境无法直连 HF。
- **解法**：所有数据/模型下载前 `export HF_ENDPOINT=https://hf-mirror.com`。`scripts/prepare_data.sh` 已内置。

### 1.2b hf-mirror 下载小模型时 401（Xet CAS）
- **现象**：`snapshot_download` 下载 `Qwen2.5-0.5B-Instruct` 时，小文件（config/tokenizer）都成功，但 `model.safetensors` 报
  `RuntimeError: Task error: File reconstruction error: CAS Client Error: HTTP status client error (401 Unauthorized), domain: https://cas-server.xethub.hf.co/...`。
  同样的方式下 `Qwen2.5-1.5B-Instruct` 却成功（先完成）。
- **原因**：`huggingface_hub` 1.33 默认启用 Xet 传输；`hf-mirror.com` 只代理常规 HTTP，不代理 Xet CAS 的上传/重建接口，重定向到 `cas-server.xethub.hf.co` 后未认证 → 401。
- **解法**：`export HF_HUB_DISABLE_XET=1` 强制走常规 HTTP 下载（`scripts/prepare_data.sh` 已内置）。重跑后补下缺失的 `model.safetensors` 即可（脚本按 `.done` 标记跳过已完成项）。

### 1.3 系统没有 nvcc / CUDA toolkit
- **现象**：`nvcc --version` → command not found；`/usr/local/cuda-12.1`、`cuda-12.8` 存在但不是 13.x。
- **原因**：驱动是 580.178.04（CUDA 13.0），但系统未装对应 toolkit。
- **解法**：**不需要**系统 toolkit。torch/vLLM 的 cu13x wheel 自带 CUDA runtime（pip 会装 `nvidia-cuda-*` 包）。本项目也不源码编译 flash-attn：verl v0.9.1 的 `[vllm]` extra 不含 flash-attn，vLLM 用自身 attention backend，训练侧 `use_remove_padding=True` 走 sdpa/FlashAttention 的可用实现。若后续确实需要 flash-attn，再单独评估（W9）。

### 1.4 没有 tmux / screen
- **现象**：`tmux`、`screen` 均未安装，且 `sudo` 需要密码。
- **解法**：长任务用 `setsid nohup ... > logs/xxx.log 2>&1 &`（协议允许 nohup）。实验脚本用 `scripts/run_exp.sh` 统一落日志，训练本身靠 verl 的 checkpoint + `resume_mode=auto` 续跑。

### 1.5 官方 Docker 镜像未使用
- **原因**：Docker 存在，但镜像拉取/挂载 16GB 单卡的收益低于直接用 conda；且协议要求记录版本，conda + `pip freeze` 更容易审计。
- **解法**：用 conda（Python 3.11）+ verl 源码 tag 安装，版本锁在 `env/versions.md` / `env/requirements.lock`。

---

## 2. verl 版本差异（v0.9.1 vs 任务包 W1 文档）

任务包写作时假设的是较老的 verl（`examples/grpo_trainer/` + `main_ppo` + 固定 micro batch 的写法）。实际安装 **v0.9.1（tag，commit 1876b06d0a3e4e71e06230be10af14492ca8a75b）** 后确认的差异：

| 项 | 任务包写法 | v0.9.1 实际 | 处理 |
|---|---|---|---|
| 版本锁定 | 未指定 | `pyproject.toml` 的 `[vllm]` extra 锁 `torch==2.11.0`、`vllm==0.24.0`、`transformers==5.9.0`（cu130） | 按 pyproject 安装，`setup.py` 只有 `vllm>=0.18.0` 的宽松约束，**以 pyproject 为准** |
| GRPO-LoRA 模板 | `examples/grpo_trainer/` 下的 LoRA 示例 | LoRA 的 canonical 示例在 **`examples/tuning/lora/run_qwen3_8b_fsdp.sh`**；`examples/grpo_trainer/` 里没有 LoRA 脚本 | 以 `examples/tuning/lora/` 为模板改写 |
| LoRA + vLLM 权重同步 | `rollout.load_format=safetensors` | 还需要 **`rollout.layered_summon=True`**（README 明确列出） | 两个参数都加 |
| micro batch | `ppo_micro_batch_size_per_gpu` | 字段仍存在，但官方示例改用 `use_dynamic_bsz=True` + `ppo_max_token_len_per_gpu` | 冒烟先用文档的固定 micro batch（可复现），OOM 时再切 dynamic |
| 启动方式 | `python3 -m verl.trainer.main_ppo` | `main_ppo` 仍在，但示例默认用 `uv run --frozen ...`，且要求 `ray_kwargs.ray_init.runtime_env.py_executable` 一致 | 用 conda python 启动，显式传 `ray_kwargs.ray_init.runtime_env.py_executable=null`（= `VERL_USE_UV=0`） |
| 数据脚本参数 | 未写 | `gsm8k.py` 用 `--local_save_dir`；`--local_dir` 已废弃（会打 warning） | 用 `--local_save_dir` |
| trainer 步数 | `trainer.total_training_steps=20` | 仍支持；`total_epochs` 为默认路径 | 保持 `total_training_steps` |
| 断点续跑 | 未写 | `trainer.resume_mode` = `auto` / `disable` / `resume_path`，默认 `auto`；checkpoint 路径 `trainer.default_local_dir` | 显式 `resume_mode=auto` + 固定 `default_local_dir` |
| 训练循环 | 未写 | `trainer.use_v1: true`（v1 sync trainer）；rollout 默认 `mode: async` | 用默认，不覆盖 |
| 删除的旧参数 | — | `examples/tuning/README.md` 明确：`ppo_micro_batch_size`、`data.val_batch_size`、top-level `reward_model.*`、`actor.ulysses_sequence_parallel_size` 已废弃 | 不使用 |

> 记录时间：2026-10-20，依据 `third_party/verl` 源码（tag `v0.9.1`）逐项核对。

---

## 3. 冒烟实验实测

### 3.1 Ray 启动失败：AF_UNIX socket 路径超过 107 字节
- **现象**：`ray.init()` 抛 `OSError: validate_socket_filename failed: AF_UNIX path length cannot exceed 107 bytes: /home/angus/workspace/SelfEvolve_RLVR/logs/ray/ray/session_.../sockets/plasma_store`。
- **原因**：Ray 把 plasma store 的 unix socket 放在 `$RAY_TMPDIR/ray/session_*/sockets/` 下；本项目路径本身较长，再叠加 Ray 的 session 目录就超限。
- **解法**：`RAY_TMPDIR` 用短路径（`$HOME/raytmp`），不要放在项目深层目录。已写入 `scripts/run_exp.sh`。

### 3.2 transformers 5.9 默认 `flash_attention_2`，本机无 flash-attn
- **现象**：worker 初始化时 `ImportError: FlashAttention2 has been toggled on, but it cannot be used due to the following error: the package for FlashAttention2 doesn't seem to be installed.`
  调用栈：`transformers/modeling_utils.py -> get_correct_attn_implementation -> _flash_attn_can_dispatch`。
- **原因**：verl v0.9.1 的 `HFModelConfig` 在 `verl/workers/config/model.py:185` 把 `attn_implementation` 默认值写死为 `flash_attention_2`，只有 `override_config.attn_implementation` 能改；而本机没有（也不打算源码编译）flash-attn，见 §1.3。
- **解法**：冒烟实验显式传
  ```bash
  actor_rollout_ref.model.use_remove_padding=False
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  ```
  - `use_remove_padding=True` 的变长路径依赖 flash-attn 的 varlen kernel，因此同时关掉；
  - 只影响 actor/ref 的 HF 前向，**vLLM rollout 仍用自己的 FLASH_ATTN backend**，不影响 rollout 生成；
  - 代价：训练前向略慢、显存略高（padding 计算），W1 冒烟可接受；W9 吞吐优化时再评估装 flash-attn 或换 vLLM 侧 kernel。

### 3.3 vLLM KV cache 不足：`max_model_len` 默认用了模型上限 32768
- **现象**：vLLM 初始化报
  `ValueError: To serve at least one request with the model's max seq len (32768), (0.88 GiB KV cache is needed, which is larger than the available KV cache memory (0.09 GiB). ... estimated maximum model length is 3488.`
- **原因**：`actor_rollout_ref.rollout.max_model_len` 在 v0.9.1 中默认 `null` → vLLM 回退到模型的 `max_position_embeddings`（Qwen2.5 = 32768），按 32k 预留 KV cache；而 `gpu_memory_utilization=0.3`（约 4.9GB）扣除 1.5B 权重后只剩 0.09GB。任务包文档没有提到这个参数。
- **解法**：显式设置 `actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN))`（本次 = 1024）。这是 16GB 单卡上必须的；不是显存档位参数的替代品，OOM 时仍按文档顺序降级。

### 3.4 单卡 FSDP1 NO_SHARD 下 `layered_summon=True` 失败
- **现象**：训练开始后第一次 `actor_rollout_update_weights()` 抛
  `NotImplementedError: offload_to_cpu=True and NO_SHARD is not supported yet`
  （`torch/distributed/fsdp/_unshard_param_utils.py` ← `FSDP.summon_full_params` ← `verl.utils.fsdp_utils.collect_lora_params`）。
- **原因**：LoRA 的官方示例（8 卡）用 `rollout.layered_summon=True` 做逐层权重同步；该路径内部调用 `summon_full_params(..., offload_to_cpu=True)`。单卡时 FSDP 的 sharding strategy 是 `NO_SHARD`，PyTorch 明确不支持 `offload_to_cpu=True + NO_SHARD` 组合。
- **解法**：单卡冒烟设 `actor_rollout_ref.rollout.layered_summon=False`（默认值）。此时走 `summon_full_params(module, writeback=False)` 路径，只同步 LoRA adapter（base 权重由 `load_format=safetensors` 预加载，`base_sync_done=True`）；NO_SHARD 下参数本来就在 GPU 上，summon 是近似 no-op，代价仅是同步时多占一份完整参数。
- **影响**：多卡（W10）可以继续用 `layered_summon=True` 省显存；该参数属于硬件档位相关项，切换硬件时需要复核。

### 3.5 vLLM 0.24 `add_lora` 报 StopIteration（adapter 路径不可用）
- **现象**：第一次权重同步时 vLLM worker 崩：
  `vllm/lora/model_manager.py:798 in _create_merged_loras_inplace → first_lora = next(iter(lora_model.loras.values())) → StopIteration`，
  调用链 `verl/.../vllm_rollout/utils.py:_update_weights → self.add_lora(TensorLoRARequest)`。
- **原因**：verl 0.9.1 的 LoRA「adapter 增量同步」路径（`peft_config and base_sync_done` 分支）构造 `TensorLoRARequest` 后交给 vLLM 0.24 的 LoRA 管理器；vLLM 侧 `loras` 为空，说明 adapter 张量没被识别成任何 LoRA 层（版本间 adapter 注册接口不兼容）。这是依赖组合问题，不是我们的配置错。
- **解法**：走 verl 官方同样支持的 **merge 路径**：`actor_rollout_ref.model.lora.merge=True`。
  - 每次同步把 LoRA 合并进 base 权重再整份传给 vLLM，完全绕过 vLLM 的 LoRA 管理器；
  - 代价：每步同步 3GB 全量权重（同卡 IPC，秒级）、merge 时多一份 3GB 权重备份（在 CPU），16GB 可承受；
  - 精度上等价（LoRA 仍然只训练 adapter，merge 只是导出方式）。
- **备选**（未采用）：`actor_rollout_ref.actor.strategy=fsdp2` + adapter 路径；或升级 vLLM —— 协议禁止擅自升级核心依赖，先记录。

### 3.6 verl v0.9.1 的 v1 trainer 在「无 critic 的 GRPO」下必崩（上游 bug）
- **现象**：第 1 步训练时 `_compute_advantage` 抛
  `ValueError: Fields ['values'] are not ready in all the requested keys: missing, or written by kv_update(finalize=False) and not finalized yet`
  （`transfer_queue/interface.py:kv_batch_get` ← `verl/trainer/ppo/v1/trainer_base.py:1761`）。
- **原因**：v1 trainer（`trainer.use_v1=True`，v0.9.1 默认）的 `_compute_advantage()` 硬编码
  `fields = [..., "values"]`，而 `values` 只有 critic worker 会写；`adv_estimator=grpo` 时 `need_critic()=False`，没有 critic → 该字段永远缺失。
  同类问题也存在于 `_compute_metrics()`。这是上游代码对「critic-free GRPO」的遗漏，不是我们的配置问题（在 16GB 上给 GRPO 加一个 1.5B critic 也不现实）。
- **解法**：`trainer.use_v1=False` 走 legacy trainer（`main_ppo_v0.py` + `ray_trainer.py`），它按 `self.use_critic` 分支处理，GRPO 不需要 `values`。
  - 两个 trainer 共用 `verl/workers/engine_workers.py` 的 `ActorRolloutRefWorker`，所以 LoRA merge、权重同步、checkpoint 逻辑不受影响；
  - 代价：会打印 deprecated 警告；W2 起如需 v1 的新特性（async rollout 等），需要等上游修复或自行 patch（patch 前要先请示，且不能改评测逻辑）。

### 3.7 冒烟实验实测结论（已跑通）

最终可用组合（`configs/EXP-W01-01-smoke.sh`，已提交）：

| 参数 | 值 | 备注 |
|---|---|---|
| `trainer.use_v1` | `False` | 规避 §3.6 的 v1 bug |
| `actor_rollout_ref.model.lora.merge` | `True` | 规避 §3.5 的 vLLM LoRA bug |
| `actor_rollout_ref.model.override_config.attn_implementation` | `sdpa` | 无 flash-attn，见 §3.2 |
| `actor_rollout_ref.model.use_remove_padding` | `False` | 同上 |
| `actor/ref.fsdp_config.model_dtype` | `bf16` | **OOM 的真正解法**，见 §3.8 |
| `rollout.max_model_len` | `1024` | 见 §3.3 |
| `rollout.layered_summon` | `False` | 见 §3.4 |
| `rollout.load_format` | `safetensors` | LoRA base 权重预加载 |
| `rollout.gpu_memory_utilization` | `0.3` | 0.25 起不来 |
| `ppo_micro_batch_size_per_gpu` | `2` | 文档初值即可（改 bf16 后） |

实测数字（Qwen2.5-1.5B-Instruct、GSM8K、train_batch 16、rollout.n 4、max_resp 512）：

- 单步 23.2s：`update_actor` 12.3s（53%）、`gen` 3.7s、`old_log_prob` 3.5s、`ref` 2.0s、`update_weights` 1.6s；
- 吞吐 ~980 tok/s；actor 进程峰值显存 9.28 GiB（allocated）/ 9.53 GiB（reserved）；vLLM 进程约 3.9 GiB；
- 验证集 `acc/mean@1`：step 10 = 0.5550、step 20 = 0.7081（仅冒烟，不作精度结论）；
- checkpoint 每个 3.6GB（actor 全量 + optimizer），`resume_mode=auto` 从 `global_step_10` 续跑成功（加载 model/optim/rng/lr_scheduler，global_step 从 11 继续）。

### 3.8 OOM 根因：FSDP 默认 `model_dtype=fp32`
- **现象**：`model.lora.merge=True` 的权重同步阶段（`restore_base_model_weights` 的 `param.data.copy_(backup[name].to(param.device))`）OOM：
  `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 892.00 MiB ... 15.57 GiB total`；此时 micro batch 已经从 2 降到 1，几乎无改善。
- **原因**：`verl/trainer/config/engine/fsdp.yaml` 的 `model_dtype` 默认是 **fp32**，1.5B 参数即 6.2GB；`backup_base_model_weights` 把参数 clone 到 CPU 再 `.to(device)` 恢复，最大的 embedding 在 fp32 下正好是 933MB（≈892MiB），一次性分配必然失败。任务文档要求的「训练统一用 BF16」也说明这里应该显式设 bf16。
- **解法**：`actor_rollout_ref.actor.fsdp_config.model_dtype=bf16` + `actor_rollout_ref.ref.fsdp_config.model_dtype=bf16`（官方 merge 示例 `examples/tuning/lora/run_qwen3_8b_merge_fsdp.sh` 正是这么写的）。改完参数占用从 6.2GB 降到 3.1GB，micro batch 恢复文档初值 2 也能跑。
- **教训**：OOM 先看 dtype / 权重副本，再降 batch；文档的降级阶梯是在「dtype 已正确」的前提下才有效。

---

## 4. W2：官方 baseline 考证与复现差异

### 4.1 官方 77.9 的脚本/日志在哪里

任务包要求"在 verl 文档 Algorithm Baselines 找到 77.9 分对应的脚本和超参"。实际考据结果：

| 项 | 内容 |
|---|---|
| 表格 | `docs/algo/baseline.md`：`Qwen/Qwen2.5-1.5B-Instruct \| GRPO-LoRA \| 77.9` |
| 表格加入 commit | `368dff726bdef15d292270806a1f641d5600039d`（PR #2083，2025-06-18） |
| 参考脚本 | `examples/tuning/1.5b/qwen2-1.5b_grpo-lora_1_h100_fsdp_vllm.sh`（同一 commit） |
| 实跑日志 | `eric-haibin-lin/verl-data` 的 `gsm8k/Qwen2.5-1.5B-bsz64_2-prompt512-resp1024-lorarank32-score0.779.log` |
| 日志现状 | 该文件已从 `experiments` 分支删除（当前只有 8 个 gsm8k 日志）；本次通过 **PR #3 的 ref**（`bcff590`）取回，存 `third_party/refs/baseline_qwen2.5-1.5b.log`（不入库） |

**从日志落盘 config 读到的真实超参**（以日志为准，不以文件名猜）：

```
train_batch_size=64, rollout.n=5, total_epochs=1 (116 steps)
lr=3e-5, lora_rank=32, lora_alpha=32, target=all-linear
kl_loss_coef=0.001 (low_var_kl), entropy_coeff=0.001
max_prompt_length=512, max_response_length=1024
2×H100, rollout.gpu_memory_utilization=0.1, TP=2
final val: val-core/openai/gsm8k/reward/mean@1 = 0.77938 @ step 116
```

**文件名陷阱**：`bsz64_2` 的 `_2` 是 **2 张卡**（1.5B/3B 用 2 卡，7B 以上用 8 卡），不是 `rollout.n=2`。
实际 `rollout.n=5`；有效 batch = 64 prompts × 5 = 320 sequences/step。若按 n=2 复现，有效 batch
只有 128，与官方不可比。

### 4.2 v0.9.1 上的字段差异（W2 实际使用）

| 官方日志字段 | v0.9.1 处理 |
|---|---|
| `actor_rollout_ref.actor.ppo_micro_batch_size` | 已废弃，改 `..._per_gpu` |
| `data.val_batch_size` | 已废弃；validation 用 `train_batch_size` |
| `VLLM_ATTENTION_BACKEND=XFORMERS` | vLLM 0.24 不设；用内置 FLASH_ATTN |
| `use_shm=True` | 保留可用，但单卡无收益，未开 |
| 2×H100 的 `layered_summon=True` | 单卡 NO_SHARD 不支持，设 `False`（§3.4） |
| 官方 `ppo_micro_batch_size=64`（每卡 64 seq） | 16GB 上必须 1；这是硬件适配，不改有效 batch |

### 4.3 4080 上的系统参数偏离（不改数学）

| 参数 | 官方 | 4080 | 理由 |
|---|---|---|---|
| `test_freq` | 5 | 20 | 只影响监控频率；省约 1h 验证时间 |
| `rollout.max_num_batched_tokens` | 1536 | 8192 | 官方为 2×80GB 保守值；纯调度参数 |
| `ppo_micro_batch_size_per_gpu` | 64 | 1 | 16GB + 1024 response |
| `rollout.gpu_memory_utilization` | 0.1 | 0.30 | 16GB |
| LoRA 同步 | adapter/layered_summon | merge 全量 | §3.5 |

### 4.4 配置变量名冲突（已修复，留档）

- **现象**：第一次 20 步校准的进程命令行里出现 `data.max_response_length=512`、
  `rollout.max_model_len=1024`、`log_prob_micro_batch_size_per_gpu=2`，与 baseline 设计不符。
- **原因**：`configs/EXP-W02-01-baseline-gsm8k.sh` 先 `source configs/hw/4080.env`，而该文件给 W1 冒烟
  export 了 `MAX_PROMPT_LEN=512` / `MAX_RESPONSE_LEN=512` / `LOGPROB_MICRO_BS=2`；
  配置里的 `${MAX_RESPONSE_LEN:-1024}` 因此永远取到 512。
- **解法**：新增 W2 专用变量 `W02_PPO_MICRO_BS` / `W02_LOGPROB_MICRO_BS`（硬件档位）与
  `BASELINE_MAX_PROMPT_LEN` / `BASELINE_MAX_RESPONSE_LEN`，不再复用 W1 变量名；启动日志打印
  `prompt/response/micro/util` 四项，便于开跑前核对。
- **代价**：一次无效的 wandb run（`eu4gz4yj`，约 1 步，已标注为 aborted）；checkpoint 目录已清空。
- **教训**：`source` 硬件档位后，所有 `${VAR:-default}` 都可能被档位变量"注入"；实验级参数必须用
  独立命名空间，不能依赖默认值。

### 4.5 评测卡在 huggingface.co：HF_ENDPOINT 设置太晚

- **现象**：基座评测在 vLLM 初始化完成后卡住，日志反复
  `'timed out' thrown while requesting HEAD https://huggingface.co/datasets/HuggingFaceH4/MATH-500/...`
  （重试 5 次）；GPU 显存已占用但 util 0%。
- **原因**：`eval/harness/data.py` 里 `os.environ.setdefault("HF_ENDPOINT", ...)` 是在 `load_dataset` 前执行，
  但 `run_eval.py` 先 import 了 vLLM/transformers，`huggingface_hub` 在 import 时已把 `ENDPOINT`
  固化成 `https://huggingface.co`；之后再改 env 不生效。
- **解法**：三层同时设置，确保在 import 链之前：
  1. `eval/run_eval.py` 在 import transformers/vllm 之前设置 env；
  2. `eval/harness/data.py` 模块顶层设置；
  3. `scripts/run_eval.sh` / `scripts/w2_chain.sh` 在启动 python 前 export。
  另加 `HF_HUB_DISABLE_XET=1`（W1 §1.2b 的 401 问题）。
- **代价**：一次卡住的评测（约 10 min GPU）与一次手动重启；checkpoint 不受影响。
- **教训**：HF 镜像/endpoint 必须在"任何 import huggingface_hub 的库"之前设置；脚本层 export 最稳。

### 4.6 判分器 bug：`#### 460 ####` 被误判为空答案

- **现象**：初版 GSM8K 结果里 37/1319 条 `extracted=""`；抽查发现模型输出形如
  `... #### 460 ####`（重复结尾标记）。这些样本的 `correct` 多数仍为 True（全文 parse 兜底），
  但 3 条被误判；MATH-500 上影响更大（0.504 → 0.564）。
- **原因**：旧 `extract_predicted_answer` 取"最后一个 `####` 之后"的文本；重复标记时该文本为空，
  候选列表里也没有正确的答案片段；math_verify 对带尾部 `####` 的全文 parse 有时失败。
- **解法（单独 commit `886f77b`）**：
  - 按 `####` 切分，取**最后一个非空片段**；对所有非空片段（含 `\boxed{}` 包装）分别 parse，任一 verify 命中即算对；
  - 新增 `eval/rescore.py`，用保存的生成结果重打分，不重跑模型；原始 `summary.json`/JSONL 保留；
  - 更新单测（49 个全过），`results_master.md` 同时记录修复前后分数。
- **影响**：GSM8K 0.7635 → **0.7665**；MATH-500 0.5040 → **0.5640**；基座 MATH 不变（0.552）。
- **教训**：生成式评测的"答案抽取"必须在真实输出上做分布检查（特别是模型学到的格式标记可能重复）；
  只看聚合分数发现不了 2.8% 的空抽取。

### 4.7 merge 一致性：权重 bit-identical，但 runtime adapter 的生成会分叉

- **现象**：`merge_lora.py` 的 3 条固定文本 logits max|Δ|=0；但 vLLM 里 PEFT runtime adapter 与 merged
  模型跑同样 20 题 greedy，只有 1/20 文本完全一致（共同前缀中位数 235 字符，准确率 14/20 vs 15/20）。
- **原因**：vLLM 的 LoRA runtime kernel 与 merged 权重的 kernel 浮点累加顺序不同；自回归生成里
  一个 token 的分叉会级联到整段。权重本身没有错：`peft.merge_and_unload()` 后
  **339/339 参数与 merged 模型 bit-identical**（max|Δ|=0，missing=0），同一进程内
  `unloaded` vs `merged` 的 20 条 GSM8K prompt logits 也是 **max|Δ|=0**（`eval/check_merge.py`）。
- **附带发现**：HF CPU bf16 的 **runtime adapter** forward 跨进程有非确定性（同一权重 logit 能差 0–10），
  所以"用 PEFT runtime 与 merged 比 logits"不是可靠判据；必须先 `merge_and_unload()` 再比权重/同进程 logits，
  或直接用 vLLM 的 merged 模型出数。
- **结论**：所有上报分数用 **merged** 模型；merge 正确性由权重级检查保证。详见
  `reports/week02_summary.md` 的"merge 一致性"。
- **教训**：生成级一致性对 kernel 差异极其敏感，不要用它判断权重 merge 是否成功。

### 4.8 W2 profiling 的两个失败点（保留记录，不静默跳过）

- **`EXP-W02-02` util=0.25**：vLLM 初始化失败（KV cache 不足）。16GB 下
  `gpu_memory_utilization=0.25` 不足以同时放 1.5B 权重（~3.1GB）+ 1536 总长的 KV。
  0.30 是可用下限，0.35 只比 0.30 快约 2%。
- **`EXP-W02-04` micro=4**：`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.39 GiB.
  GPU 0 has a total capacity of 15.57 GiB of which 839.88 MiB is free.`
  1024 response × 4 条序列的激活放不下。micro=2 是 16GB 上的实际上限（峰值 11.91 GiB）。
- 两个失败点都写进了 `reports/profiling_baseline_4080.md` §5，作为参数边界的一部分。

### 4.9 文档错记：训练内 validation 的采样温度（已更正）

- **现象**：`eval/datasets.md §5` 与 `reports/results_master.md` 的 W1 两行把 verl 内置验证循环
  记成 `temperature=1.0`，而 `reports/week02_summary.md` 的风险提示写"两者都是 greedy"。
  同一件事在三处有两种说法，而这正是"harness 76.65 vs 训练内 75.82"差异解释的依据。
- **原因**：verl 的 `rollout.yaml` 里有两组采样参数——**顶层** `temperature: 1.0 / top_p: 1 / n: 1`
  是**训练 rollout** 用的，另有 `val_kwargs:` 一组是**验证**用的。初稿读到了顶层的 1.0。
- **事实**（已锁定的 verl commit `1876b06`，`verl/trainer/config/rollout/rollout.yaml`）：
  `val_kwargs: temperature: 0, top_p: 1.0, top_k: -1, n: 1, do_sample: False`。
  本项目的 `configs/*.sh` 全部没有覆盖 `actor_rollout_ref.rollout.val_kwargs`
  （`grep -rn 'val_kwargs\|do_sample' configs/` 为空），所以训练内 validation 是 **greedy**。
- **结论**：两套口径的差异**只在判分器与答案抽取**（verl 内置 GSM8K rule reward 只比对 `####`
  后的数值 vs Math-Verify + 本项目抽取器），**不在采样温度**。
- **解法**：`eval/datasets.md §5` 改为对照表并附证据；`results_master.md` 的 W1 两行采样参数更正，
  并在说明里留更正记录。`week02_summary.md` 的表述本来就是对的，补上证据来源。
- **教训**：同一个参数在上游配置里出现两次（训练/验证各一套）时，必须写清引用的是哪一处 + 文件 +
  commit。文档之间互相矛盾时，先去上游配置取证，不要选一个"看起来合理"的。

---

## 5. W3：严格 verifier 与上游/文档的差异

### 5.1 自定义奖励的配置键

- **任务包写法**：`custom_reward_function.path` / `custom_reward_function.name`（顶层）。
- **v0.9.1 实际**：`reward.custom_reward_function.path` / `.name`
  （`verl/trainer/config/reward/reward.yaml`；顶层同名键只出现在 `legacy_reward_impl.yaml`，
  `get_custom_reward_fn()` 读的是 `config.reward.get("custom_reward_function")`）。
- **调用约定**（`verl/experimental/reward_loop/reward_manager/naive.py`）：关键字调用
  `compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)`；
  `verl/trainer/main_eval.py` 则是位置调用。返回 float 或 dict 都支持，dict 时取 `result["score"]`，
  其余键进 `reward_extra_info`（落 wandb / parquet）。
- **处理**：`src/selfevolve/rewards/strict_reward.py` 的签名
  `compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)` 两种调用都兼容；
  配置用 `reward.custom_reward_function.*`（见 `configs/EXP-W03-01-strict-reward-smoke.sh`）。
- **自带并行**：`reward.num_workers=8`，`RewardLoopManager` 起 8 个 reward worker 分片——不需要自建并行。

### 5.2 Math-Verify 的超时在子线程里不可用

- **现象**：`math_verify.parse(..., parsing_timeout=5)` 在非主线程抛
  `ValueError: Math-Verify 'parse' function doesn't support threaded environment due to usage of signal.alarm()`。
  Windows 上则是每次调用 `multiprocessing.Process`（官方 docstring 自己说 "huge performance penalty"）。
- **原因**：`math_verify/utils.py::timeout` 在 POSIX 用 `signal.alarm`（只对主线程有效），
  Windows 回退到多进程。
- **训练侧为什么会踩到**：verl 的 `NaiveRewardManager.run_single` 用
  `self.loop.run_in_executor(None, ...)` 调 `compute_score`，**跑在线程池里**，不是主线程。
- **解法**：严格 verifier 自己管超时——`parse(parsing_timeout=None)`、`verify(timeout_seconds=None)`，
  外层 `_run_with_timeout`：POSIX 主线程用 `signal.setitimer`，其余情况用进程池 +
  `future.result(timeout=5)`；进程池基础设施出错时退化为同步执行（判定结果不变，只是没有超时保护）。
- **附带**：关掉上游超时会触发它每次打印 "Timeout is disabled ..." 到 stdout，已在 import 时把
  `math_verify.parser/grader.TIMEOUT_WARNING_SHOWN` 置位，保持训练日志干净。

### 5.3 Math-Verify 的解析行为（实测，0.9.0）

| 输入 | `parse` 结果 | 影响 |
|---|---|---|
| 裸 `460` | `[]` | gold 必须包成 `\boxed{460}`（W2 已处理，严格 verifier 也这么做） |
| `\dfrac{3}{4}`（未包裹） | `[]` | 归一化层把 `\dfrac→\frac` |
| `3.5\times 10^{3}`（未包裹） | `[10]`（只取到指数） | 归一化层自己算科学计数法 |
| `1e3` | `[1*E*3]` | 同上 |
| `3 or 5` | `[5]`（只取一个） | 多答案必须自己判，否则会被"任意匹配"混过去 |
| `\boxed{3} and \boxed{5}` | `[{3,5}]` | 同上 |
| `(3,5)` | `[\text{Interval.open}(3,5)]`（当成开区间） | 两侧同样处理时不影响相等性；但"坐标"语义丢失，已记录 |
| `50\%` | `[50*(1/100), '50\%']`，而 `verify` 既能对 0.5 也能对 50 | 太宽松：归一化层统一把 `%` 当**单位**去掉（`50\% → 50`），与数据集 gold 的裸数字口径一致 |
| `\begin{pmatrix} a \\ b \end{pmatrix}` | 解析失败 | 已知限制，见 `reports/verifier_validation.md §6` |

### 5.4 非交互 ssh 启动训练没有 conda python

- **现象**：`setsid nohup bash scripts/run_exp.sh EXP-W03-01-...` 立刻退出，
  `configs/EXP-W02-01-baseline-gsm8k.sh: 行 75: python: 未找到命令`，`exit code: 127`。
- **原因**：conda 是在 `~/.bashrc` 里初始化的，非交互 ssh 的 shell 不读它，`PATH` 里没有 `python`。
  （W2 的实验是从交互式 shell 里启动的，所以没暴露。）
- **解法**：
  ```bash
  source /home/angus/miniconda3/etc/profile.d/conda.sh && conda activate selfevolve && \
  setsid nohup bash scripts/run_exp.sh <EXP-ID> > logs/<name>.out 2>&1 < /dev/null &
  ```
- **代价**：一次空跑（0 GPU·h，进程在 import 前就退出了）。

### 5.5 用 PowerShell 写 UTF-8 文件会坏掉

- **现象**：`Add-Content experiments/registry.csv "<中文行>"` 之后，文件不再是合法 UTF-8
  （Python 读会报 `invalid UTF-8`，`file` 显示 `Non-ISO extended-ASCII`），那一行中文也变成乱码。
- **原因**：Windows PowerShell 5.1 的 `Add-Content` 默认用系统 ANSI 代码页（GBK）写文件，
  而仓库里所有文件都是 UTF-8。
- **解法**：凡是写仓库里的文本文件，一律用项目 venv 的 python 或编辑器工具（`Path.write_text(..., encoding="utf-8")`），
  不要用 `Add-Content` / `Out-File`。已把坏行用 python 重写并单独 commit。

### 5.6 训练奖励第一次调用偏慢（29 ms）是进程池冷启动

- **现象**：冒烟第 1 步 `timing_s/agent_loop/compute_score/mean ≈ 29 ms`、`max ≈ 0.53 s`，
  比离线测的 1.05 ms 高了近 30 倍。
- **原因**：8 个 reward worker 各自第一次调用时才创建进程池（spawn + import math-verify），
  这批固定开销被摊到第 1 步的 320 个样本上。
- **结论**：后续步回落到毫秒级；320 样本 × 1.05 ms ÷ 8 workers ≈ 0.04 s/step（0.02%），不构成瓶颈。
  记录在此，避免下次把首次开销误判成"判分器太慢"。

### 5.7 训练内 validation 的口径随奖励函数改变

- **现象**：W3 冒烟的 `val-core/.../reward/mean@1` 与 W2 同 step 的数不可直接比较。
- **原因**：verl 的训练内验证也用同一个 `custom_reward_function`（`trainer.val_before_train` /
  `test_freq` 走 `_validate`，里面调用同一个 `compute_score`）。W2 用的是 verl 内置 GSM8K rule reward
  （要求 `#### 数字`），W3 是严格 verifier（含结论句兜底），两者对同一批验证集的打分不同。
- **处理**：W3 冒烟的 `critic/score/mean` 与 W2 对照时，必须说明这是"不同判分器下的 reward"；
  评测口径的对照一律用 `outputs/eval/*` 的 harness 结果（同一 verifier 重打分）。

### 5.8 严格 verifier 的"多答案"判定与 GSM8K 的 `####` 约定

- **现象/决策**：模型可能在一条输出里写多个 `####`（先写草稿值再补全，或 `#### 15` 后跟 `-15`）。
- **决策**：多个 `####` **不**判 `multiple_answers`，按"最后一个 `####` 之后是最终答案"取最后一个
  （与 verl 内置判分、GSM8K 官方约定一致）；但多个 `\boxed{}` 互不一致时判 `multiple_answers`
  （任务包明确要求）。
- **代价**：`#### 15\n-15` 这种"最后一行是另一个值"的写法会取到 `-15`。抽样 420 条里出现 1 条，
  已记入验证报告 §6 的已知限制。

### 5.9 后台"等训练结束"的脚本自匹配 `pgrep -f`

- **现象**：冒烟训练（12:29 已 exit 0）结束后 4 个多小时，本地后台任务仍显示 running，
  一直没收到"训练结束"的通知；最后是手动轮询日志才发现的。
- **原因**：等待脚本长这样
  `ssh home "while pgrep -f 'verl.trainer.main_ppo' > /dev/null; do sleep 20; done; ..."`。
  `pgrep -f` 匹配**整条命令行**，而 `bash -c` 自己的命令行里就含 `verl.trainer.main_ppo` 这个字符串，
  所以永远至少匹配到自己 → 循环不会退出。`ps` 里那条 `bash -c while pgrep ...` 就是它自己。
- **解法**（任选其一）：
  1. 方括号技巧：`pgrep -f '[v]erl.trainer.main_ppo'`（自身命令行里是 `[v]erl...`，不匹配）；
  2. 启动时记下 PID，用 `kill -0 "$PID"` 判断；
  3. 让训练脚本在结束时写 `.done` 标记文件，等待方轮询文件（与协议 §8 的"每个阶段写 .done"一致）。
- **影响**：无副作用（只是每 20 s 一次 sleep，0 CPU，GPU 早已空闲），但会让"等长任务"的信号失效；
  以后后台等待一律用上面三种写法之一。

### 5.10 verl v0.9.1 自带 rollout 落盘，不需要在 reward 里按 1% 抽样

- **背景**：W4 任务包写的是"如果安装的 verl 版本支持 rollout 或验证生成的落盘功能，就直接用；
  否则在自定义 reward 函数里按 1% 的比例抽样写 JSONL"。
- **实际**：v0.9.1 有 `trainer.rollout_data_dir`（`verl/trainer/config/ppo_trainer.yaml` 第 162 行）。
  每个训练步在 `ray_trainer.py::_log_rollout_data` 里把 decode 后的 `input` / `output` / `gts` /
  `score` 和 reward dict 的全部额外键，经后台线程写成 `<global_step>.jsonl`，退出时 drain。
- **结论**：W4 直接用原生落盘（全量 160 seq/step），不做 1% 抽样。数据在
  `outputs/w04/rollouts/<EXP>/<step>.jsonl`。
- **注意**：只有 reward dict 里返回的键会进 dump；**每条样本必须键集合一致**，否则 verl 组
  `non_tensor_batch` 时数组长度对不上。`loose_common.build_result` 就是为此保证所有键都存在。

### 5.11 训练循环不把 reward dict 的额外键写进 wandb（只有 validation 会）

- **现象**：W4 需要把 reward gap、数字个数、boxed 率等检测指标记进 wandb；这些键在
  `_validate()` 的 `reward_extra_infos_dict` 里会被 `val-aux/...` 消费，但训练循环只做
  `batch.non_tensor_batch.update(...)`（`ray_trainer.py` 约 1635 行），没有 `metrics.update`。
- **解法**：训练结束后 `analysis/hack_metrics.py` 从 rollout dump 聚合逐步指标，再用
  `wandb.init(id=<run_id>, resume="allow")` 回填到同一个 run（键前缀 `hack/`、`eval/`）；
  指标 JSON 同时落盘作为第一手证据。run id 从 `logs/<EXP>_*.log` 的 `/runs/<id>` 行解析。
- **影响**：不改 verl 源码、不改评测代码；回填失败只影响展示，不影响分析结论。

### 5.12 `MAX_*_LEN` / `LOGPROB_MICRO_BS` 被 `configs/hw/*.env` 的同名变量静默覆盖

- **现象**：W4 配置里写 `MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-1024}`，实际按 512 跑了 2 步；
  命令行传 `LOGPROB_MICRO_BS=8`，实际生效 2。
- **原因**：`source configs/hw/4080.env` 已经 `export` 了 W1 冒烟用的 `MAX_PROMPT_LEN=512`、
  `MAX_RESPONSE_LEN=512`、`LOGPROB_MICRO_BS=2`。config 里的 `:-` 默认值只对"未设置"生效。
  W2 baseline 用 `BASELINE_MAX_*` 前缀避开过同一个坑，W4 又踩了一次。
- **解法**：实验专用变量一律带前缀（`W04_MAX_RESPONSE_LEN`；临时覆盖用 `W04_MICRO_BS` /
  `W04_LOGPROB_MICRO_BS` 再映射到 hw env 的 `W04_*` 键）。
- **影响**：两次 2 步空跑（各约 2 分钟，GPU 0.06 h 量级），无数据污染（走的是独立 smoke 目录）。

### 5.13 Qwen2.5-0.5B + response 1024 在 4080 16GB 上的 micro batch 实测

- **背景**：0.5B 参数只有 1.5B 的 1/3，直觉上可以放大 micro batch；但 response 1024 时
  `use_remove_padding=False`，每条序列都 pad 到 1024，激活和 logits 并不小。
- **实测**（`EXP-W04-00` 2 步冒烟）：
  1. actor micro=4（20 seq/微批）→ `update_actor` OOM：还差 1.95 GiB，只剩 1.94 GiB；
  2. actor micro=2、logprob/ref micro=8 → `compute_log_prob` OOM：一次要 4.99 GiB，
     因为 logits 是 `8 × 1536 × 151936 × 2B ≈ 3.7 GiB`，再加 softmax 中间量；
  3. **actor micro=2、logprob/ref micro=2 通过**：稳定步时 45–53 s，actor 峰值
     9.69 GiB allocated / 12.15 GiB reserved，吞吐约 1350–1500 tok/s。
- **结论**：W4 固定 `W04_PPO_MICRO_BS=2`、`W04_LOGPROB_MICRO_BS=2`；换 3090（24GB）后可放大并重测。
- **注意**：由于 `use_remove_padding=False`，步时按 1024 token/序列计算，**不随平均响应长度下降**；
  L4 的"长度暴涨"主要增加 `gen` 时间，不会让 `update_actor` 成倍变慢。

### 5.14 JSONL 不能用 `str.splitlines()` 切：模型输出里有 U+2028

- **现象**：W4 分析训练到一半的 rollout dump 时报
  `75.jsonl: 15/170 malformed lines`，但 `wc -l` 是 160 行；同一批数据里偶发
  `json.decoder.JSONDecodeError: Unterminated string`。
- **原因**：Qwen 的输出里会带 U+2028（LINE SEPARATOR）/ U+0085 等 Unicode 行分隔符，
  `json.dumps(..., ensure_ascii=False)` 原样写出；`str.splitlines()` 把这些字符也当换行，
  于是一条完整 JSON 被拆成两半（15 条"坏行" = 7 条被拆开的记录 + 1 条半行）。
- **解法**：所有 JSONL 解析统一用 `text.split("\n")`，不用 `splitlines()`；分析脚本对
  极少量半行（训练仍在写最新步）容忍并告警，坏行超过 5% 直接报错，避免静默少算样本。
- **影响**：只影响"边训练边分析"的调试；训练数据和最终分析不受影响。
  已同步修 `analysis/hack_metrics.py` 与 `analysis/select_hack_examples.py`。
