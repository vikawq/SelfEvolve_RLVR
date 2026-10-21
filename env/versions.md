# 环境版本记录（W1 实测）

采集时间：2026-10-20 20:15（Asia/Shanghai）。全部为真实运行输出，未做修饰。

## 1. 主机

| 项 | 值 |
|---|---|
| 主机名 | bigdog01 |
| OS | Ubuntu 24.04.3 LTS，kernel 7.0.0-38-generic x86_64 |
| CPU | 24 逻辑核 |
| 内存 | 125 GiB（采集时空闲 120 GiB） |
| 磁盘 | `/` 183G（剩 18G，**91% 已用**）；`/home` 1.6T（剩 838G）；`/dev/shm` 63G |
| 登录 | 本地私有主机，登录方式不入库（见 .env / 本地 ssh config） |
| 工作目录 | `/home/angus/workspace/SelfEvolve_RLVR` |

> 注意：根分区仅剩 18G，所有缓存（pip / HuggingFace / TMPDIR）与 conda env 都放在 `/home` 下，避免撑爆根盘。

## 2. GPU

| 项 | 值 |
|---|---|
| 型号 | **NVIDIA GeForce RTX 4080** |
| 显存 | 16376 MiB（16GB 档） |
| 架构 / 计算能力 | Ada / **sm_89**（compute_cap 8.9） |
| 驱动 | **580.178.04**（`nvidia-smi` 报告 CUDA Version 13.0） |
| PCIe | Gen 1（采集时降频待机，不是上限） |
| 训练精度 | BF16（两张卡统一） |

系统 toolkit：`/usr/local/cuda` → 实际为 12.1 / 12.8，**不用于本项目**。PyTorch/vLLM 的 cu130 wheel 自带 CUDA runtime（pip 安装 `nvidia-*` 包），系统不需要 nvcc。

## 3. Python 环境

| 项 | 值 |
|---|---|
| 环境管理 | miniconda 25.9.1（`/home/angus/miniconda3`） |
| env 名 | `selfevolve` |
| Python | 3.11.17 |
| pip | 26.2.1 |

## 4. 核心依赖（实测）

| 包 | 版本 | 说明 |
|---|---|---|
| torch | **2.11.0**（`torch.version.cuda=13.0`） | `cuda.is_available()=True`，device = RTX 4080 |
| torchvision | 0.26.0 | verl `[vllm]` extra 锁定 |
| torchaudio | 2.11.0 | 同上 |
| vllm | **0.24.0** | 单卡离线生成已验证（见下） |
| verl | **0.9.1** | 源码 tag `v0.9.1`，commit `1876b06d0a3e4e71e06230be10af14492ca8a75b`（2026-09-20） |
| transformers | 5.9.0 | 与 verl 0.9.1 锁定 |
| tokenizers | 0.22.2 | |
| ray | 2.59.0 | 单节点 |
| peft | 0.21.2 | LoRA |
| accelerate | 1.15.0 | |
| tensordict | 0.10.0 | |
| flashinfer-python | 0.6.12 | vLLM 采样/注意力依赖 |
| triton | 3.6.0 | |
| numpy | 2.3.5 | |
| wandb | 0.31.0 | 已登录（`~/.netrc` 有 api.wandb.ai） |
| datasets | 5.1.0 | GSM8K 预处理 |
| pyarrow | 26.0.0 | parquet |
| math-verify | **0.9.0** | W2 评测 harness 判分器（`eval/harness/scoring.py`） |
| latex2sympy2_extended | 1.11.0 | math-verify 依赖 |
| matplotlib | 3.11.2 | W2 profiling 出图（`analysis/parse_profile.py`） |

- **flash-attn 未单独安装**：vLLM 0.24 自带 attention 实现，实测加载 `Qwen2-1.5B-Instruct` 时选中的后端是 `FLASH_ATTN`（vLLM 内置版），因此本节与任务文档「安装与该 verl 版本匹配的 flash-attn」的差异为：**不单独编译 flash-attn**，理由见 `docs/troubleshooting.md` §1.3。
- `deepspeed` 未安装：verl 0.9.1 默认 FSDP 路径，不需要。

完整锁定见 `env/requirements.lock`（`pip freeze --all`，**264** 个包；W2 新增 math-verify / latex2sympy2_extended / matplotlib 及其依赖）。

## 5. 安装命令（可复现）

```bash
conda create -y -n selfevolve python=3.11
conda activate selfevolve
mkdir -p third_party && cd third_party
git clone --depth 1 --branch v0.9.1 https://github.com/volcengine/verl.git verl
cd verl
python -m pip install -e ".[vllm]"
```

## 6. 冒烟验证记录

### 6.1 import + CUDA

```
torch 2.11.0+cu130 cuda_avail True cuda 13.0
verl 0.9.1 / vllm 0.24.0 / transformers 5.9.0 / ray 2.59.0 / peft 0.21.2
```

### 6.2 vLLM 离线生成

用本机已有的 `Qwen2-1.5B-Instruct`（临时验证 vLLM 可用性，非项目主模型）：

- 模型加载 2.98 GiB，KV cache 4.93 GiB（`gpu_memory_utilization=0.55`，`max_model_len=1024`，`enforce_eager=True`）
- 引擎初始化 42.66s，生成正常（`12+35` → `47`）
- attention backend：`FLASH_ATTN`（vLLM 内置 FlashAttention v2）

### 6.3 GSM8K

`train.parquet` 7473 行、`test.parquet` 1319 行，字段见 `docs/data_format.md`。

### 6.4 冒烟实验 EXP-W01-01-smoke（已跑通）

| 项 | 结果 |
|---|---|
| 轨迹 | phase1 step 1–10 → `global_step_10`；phase2 `resume_mode=auto` 续跑 step 11–20 → `global_step_20` |
| 结果 | 两次 launch 均 exit 0，无 OOM |
| 单步耗时 | 23.2s（`update_actor` 12.3s / `gen` 3.7s / `old_log_prob` 3.5s / `ref` 2.0s / `update_weights` 1.6s） |
| 吞吐 | ~980 tok/s |
| 显存 | actor 进程峰值 9.28 GiB allocated / 9.53 GiB reserved；vLLM 进程 ~3.9 GiB |
| 验证集 acc@1 | step 10 = 0.5550；step 20 = 0.7081（仅冒烟，不作精度结论） |
| checkpoint | 3.6GB/个；续跑日志含 `Resuming from .../global_step_10` |
| GPU 时长 | 成功轨迹 ≈0.18 GPU·h；含调试 ≈0.5 GPU·h |
| 花费 | $0（本地机器） |

### 6.5 W2 baseline 校准（`EXP-W02-01`，20 步）

| 项 | 结果 |
|---|---|
| 轨迹 | 20 步，57m28s，平均 **172.4 s/step** |
| 单步构成 | `update_actor` ~116s（67%）、`old_log_prob` ~29s、`gen` ~15s、`ref` ~11s、`reward` 1–2s、`update_weights` 1.6s |
| 吞吐 | 580–716 tok/s（320 sequences/step，response mean ~190–280 tok） |
| 显存 | actor 峰值 9.28 GiB allocated / 9.53 GiB reserved |
| 训练内验证 | step 20 GSM8K `acc/mean@1` = **0.7384**（verl 内置 rule reward，非 harness） |
| 外推 | 116 步 ≈ 4.5–5.5 h（含验证）；**< 48h**，按任务包规则无需请示，直接跑满 |
| wandb | calibration `runs/h3eaiogf`；full run `runs/hlrzg966` |

关键环境结论（详见 `docs/troubleshooting.md`）：
- FSDP 必须显式 `model_dtype=bf16`（默认 fp32 会导致 1.5B 模型在 16GB 上 OOM）
- `trainer.use_v1=False`（v1 在 critic-free GRPO 下读 `values` 字段崩溃）
- LoRA 同步走 `model.lora.merge=True`（vLLM 0.24 的 `add_lora` 与 verl adapter 路径不兼容）
- 无 flash-attn：actor/ref 用 SDPA；vLLM rollout 仍用内置 FLASH_ATTN

## 7. 待办

- 3090 到位后：`HW=3090` 跑 `EXP-W01-02-smoke-3090`，用实测值更新 `configs/hw/3090.env`，并把两卡 profiling 结果并列留档。
