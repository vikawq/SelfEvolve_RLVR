# GSM8K 数据格式

生成方式（verl v0.9.1 自带脚本）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
mkdir -p data/gsm8k
python third_party/verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

| 文件 | 行数 | 用途 |
|---|---|---|
| `data/gsm8k/train.parquet` | 7473 | 训练（GRPO rollout） |
| `data/gsm8k/test.parquet` | 1319 | **只用于评测**，不进入训练/过滤/出题提示 |

## Schema（两个文件相同，5 列）

| 列 | 类型 | 含义 |
|---|---|---|
| `data_source` | string | 固定为 `openai/gsm8k`；verl 内置 reward 按它选择打分函数 |
| `prompt` | list[dict] | chat 格式，仅一条 `role=user`；题干后追加指令 `Let's think step by step and output the final answer after "####".` |
| `ability` | string | 固定 `math` |
| `reward_model` | struct | `{"style": "rule", "ground_truth": "<最终数值>"}`；`style=rule` 表示规则判分，不看 CoT |
| `extra_info` | struct | `split`、`index`、`answer`（含 `<<...>>` 计算标注的完整 CoT）、`question`（原题干） |

注意：
- `ground_truth` 是**去掉千分位逗号后的数值字符串**（如 `"72"`、`"18"`），不是完整解答；完整解答在 `extra_info.answer`。
- `prompt` 里已经拼好了指令后缀，训练脚本不需要再拼接。
- 同一个 prompt 在 `data.train_batch_size` 内会被采样 `rollout.n` 次，用于 GRPO 的组内优势。

## 样本（原样打印自 parquet）

### train[0]

- `data_source`: `openai/gsm8k`
- `ground_truth`: `72`
- `prompt` 原文：

```
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after "####".
```

- `extra_info.answer`（完整 CoT）：

```
Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72
```

### train[1]

- `data_source`: `openai/gsm8k`
- `ground_truth`: `10`
- `prompt` 原文：

```
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn? Let's think step by step and output the final answer after "####".
```

- `extra_info.answer`：

```
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10
```

### test[0]

- `data_source`: `openai/gsm8k`
- `ground_truth`: `18`
- `prompt` 原文：

```
Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Let's think step by step and output the final answer after "####".
```

- `extra_info.answer`：

```
Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
She makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.
#### 18
```

## 红线

测试集只能用于评测，绝不能进入训练、过滤或出题提示（`02_AGENT_PROTOCOL.md` §4）。W3 之后训练奖励与评测必须共用同一严格 verifier。
