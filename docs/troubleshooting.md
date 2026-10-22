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
