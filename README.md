# SelfEvolve-RLVR

在单张消费级 GPU 上构建**可复现、可审计、跑得快**的 RLVR 自我进化后训练系统：不使用人工标注训练数据，由模型自己出题、自己解题、自我提升；同时系统研究它在什么条件下有效、什么条件下会被奖励欺骗。

> 本仓库当前进度：**W3（严格 Verifier：训练奖励与评测共用的判分器）**。RLVR = Reinforcement Learning with Verifiable Rewards。

## 环境搭建

硬件：Linux + 单张 NVIDIA GPU（当前实测 RTX 4080 16GB；3090 24GB 档位见 `configs/hw/3090.env`），内存 ≥ 64GB，`/home` 可用磁盘 ≥ 200GB。

### 1. 系统与驱动

记录在 `env/versions.md`。本项目实测环境：驱动 580.178.04（CUDA 13.0 runtime）、Ubuntu 24.04、24 核 / 125GB 内存。**不需要**系统安装 CUDA toolkit：PyTorch/vLLM 的 cu13x wheel 自带 CUDA runtime（细节见 `docs/troubleshooting.md`）。

### 2. Python 环境（conda，推荐）

```bash
conda create -y -n selfevolve python=3.11
conda activate selfevolve
# 把 pip 缓存/临时目录放到大盘上，避免撑爆根分区
export PIP_CACHE_DIR=$HOME/.cache/pip TMPDIR=$HOME/tmp
```

### 3. verl（固定 release tag）

```bash
mkdir -p third_party && cd third_party
git clone --depth 1 --branch v0.9.1 https://github.com/volcengine/verl.git verl
cd verl
# pyproject 的 [vllm] extra 锁定了配套版本（torch 2.11.0 / vllm 0.24.0 / transformers 5.9.0）
python -m pip install -e ".[vllm]"
```

`v0.9.1` 起官方示例默认用 `uv run --frozen` 启动；本仓库统一用 conda 环境里的 python，因此脚本都显式传 `ray_kwargs.ray_init.runtime_env.py_executable=null`（等价于 `VERL_USE_UV=0`）。

### 4. 数据

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 本机直连 huggingface.co 超时，走镜像
mkdir -p data/gsm8k
python third_party/verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

产物：`data/gsm8k/train.parquet`、`data/gsm8k/test.parquet`。字段说明见 `docs/data_format.md`。

### 5. 模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
for m in ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    snapshot_download(m, local_dir=f"models/{m.split('/')[-1]}")
PY
```

### 6. 冒烟实验

```bash
# 硬件档位只由 HW 一个变量切换
HW=4080 bash scripts/run_exp.sh EXP-W01-01-smoke
```

冒烟实验 `EXP-W01-01-smoke` 的配置在 `configs/EXP-W01-01-smoke.sh`，20 步 GRPO + LoRA，checkpoint 每 10 步保存。

## Baseline 与评测（W2）

```bash
# 官方 Qwen2.5-1.5B GRPO-LoRA GSM8K baseline（64 prompts × n=5，1 epoch=116 步）
HW=4080 bash scripts/run_exp.sh EXP-W02-01-baseline-gsm8k

# 一条命令评任意基座 / LoRA checkpoint（checkpoint 会先自动 merge）
bash scripts/run_eval.sh models/Qwen2.5-1.5B-Instruct
bash scripts/run_eval.sh checkpoints/EXP-W02-01-baseline-gsm8k/global_step_116
bash scripts/run_eval.sh <ckpt> --limit 20 --datasets gsm8k   # 冒烟
```

评测口径、数据集来源、AIME 的 avg@16 定义见 [`eval/datasets.md`](eval/datasets.md)；
profiling 解析见 `analysis/parse_profile.py`，报告在 `reports/profiling_baseline_4080.md`。

## 严格 Verifier（W3）

判分器只有一份实现：`src/selfevolve/verifier/math_verifier.py`。**训练奖励和评测都调它**
（协议 §4：训练奖励与评测必须共用同一严格 verifier）。

```python
from selfevolve.verifier.math_verifier import verify

r = verify("Therefore, Janet makes $18 every day.", "18")
# VerifyResult(score=1.0, extracted='18', parse_ok=True, reason='correct')
```

- 规则：只奖励正确性（1.0 / 0.0）；抽取顺序 `\boxed{}` → `####` → 收尾句 → 结论句；
  多候选、截断、无答案一律判错；单次判定 5 秒超时。
- 训练侧接线：`configs/EXP-W03-01-strict-reward-smoke.sh`
  （`reward.custom_reward_function.path=src/selfevolve/rewards/strict_reward.py`）。
- 评测侧：`eval/harness/scoring.py` 是它的薄封装；`eval/rescore.py --suffix strict` 可零 GPU 重打分。
- 验证报告：[`reports/verifier_validation.md`](reports/verifier_validation.md)
  （三判分器对比、人工核对包、已知限制）；与上游 Math-Verify 的差异记在
  `docs/troubleshooting.md §5`。

```bash
pytest -q                                   # 164 用例（含数学判分的等价形式/对抗样本/边界）
python analysis/verifier_compare.py         # 三判分器对比 + 人工核对包
python analysis/verifier_bench.py --n 500   # 单样本耗时（mean / p99）
```

## 目录结构

见 `02_AGENT_PROTOCOL.md`。核心目录：`configs/`（实验与硬件档位）、`data/`（不入库）、`src/selfevolve/`（verifier / rewards / challenger / solver / metrics）、`experiments/`（registry）、`reports/`（周报与结果汇总）、`docs/`（数据格式、指标键名、排障）。

## 设计文档

| 文档 | 回答的问题 |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | 系统分层、进程与显存拓扑、关键架构决策、信任边界 |
| [`docs/modules.md`](docs/modules.md) | 代码模块职责、接口契约、依赖规则、测试矩阵 |
| [`docs/pipeline.md`](docs/pipeline.md) | 训练单步、自我进化单轮、多轮回路、评测与实验生命周期流程 |

三份文档中的 **[已实现] / [设计] / [计划]** 标注对应当前进度，不代表全部代码都已存在。

## 复现性约定

- 每个实验有 ID（`EXP-W<周>-<序号>-<短名>`），先登记 `experiments/registry.csv` 再跑。
- 所有结果标注 GPU 型号与判分器版本；测试集不进入训练/过滤/出题。
- 环境版本锁在 `env/versions.md` + `env/requirements.lock`；与上游文档的差异记在 `docs/troubleshooting.md`。
