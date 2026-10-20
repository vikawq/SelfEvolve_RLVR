# 数据目录

`data/` 不入库（见 `.gitignore`），只保留本说明。

## GSM8K

由 verl 自带脚本从 HuggingFace `openai/gsm8k`（config `main`）生成，**只用于训练与评测，不得改写标签**：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # 本机直连 huggingface.co 超时
mkdir -p data/gsm8k
python third_party/verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

产物：

| 文件 | 说明 |
|---|---|
| `data/gsm8k/train.parquet` | 训练集，7473 条 |
| `data/gsm8k/test.parquet` | 测试集，1319 条（只用于评测） |

字段结构见 `docs/data_format.md`。
