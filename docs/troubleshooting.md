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

## 3. 冒烟实验实测（待回填）

（跑完后按「现象 / 原因 / 解法」补齐：OOM 调整、实际显存、step 耗时、可用参数）
