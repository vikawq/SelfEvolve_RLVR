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
