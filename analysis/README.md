# analysis/

画图与 profiling 解析。W2 起：

```bash
# 解析训练日志（支持同一 run 多次 launch 合并），输出 CSV/JSON/饼图/敏感度图
python analysis/parse_profile.py --logs 'logs/EXP-W02-0*.log' --out-dir reports/figures

# 也可以直接从 wandb 拉（自建实例；需要已登录）
python analysis/parse_profile.py --wandb-run <entity>/<project>/<run_id>
```

产物：
- `analysis/profile_summary.csv` / `.json`：每个 run 的阶段中位数、吞吐、峰值显存
- `reports/figures/profile_phase_pie.png`：baseline 单步耗时构成
- `reports/figures/profile_sensitivity.png`：三组参数扫描对比

口径：剔除 step 1（warmup）；含 testing / save_checkpoint 的步不计入"纯训练步"的耗时占比。
所有图必须能由 `experiments/registry.csv` 与日志/wandb 数据重建。
`profile_summary.json` 是 `reports/profiling_baseline_4080.md` 所有数字的**唯一来源**：
改报告先对照这个文件，阶段占比的分母统一取 `median_step_train_s`。

## 评测结果行

```bash
python analysis/emit_results_row.py --summary outputs/eval/<run>/summary.json \
  --date 2026-10-21 --exp-id EXP-W02-01-baseline-gsm8k \
  --model 'Qwen2.5-1.5B-Instruct（基座，无 RL）' --gpu 'RTX 4080 16GB' --wandb '—'
```

输出 `reports/results_master.md` 的 Markdown 行（分数 / stderr / std / 截断率 / 空答案率 /
判分器版本），避免手抄。W3 按严格 verifier 重评时，用它批量生成追加行。
