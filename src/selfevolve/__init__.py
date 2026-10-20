"""SelfEvolve-RLVR 核心包。

子模块规划（对应 02_AGENT_PROTOCOL.md 第 1 节）：
- verifier: 严格判分器（W3）
- rewards: verl 自定义奖励函数，含宽松判分器（W4）
- challenger: 出题与 Challenger 奖励（W6）
- solver: 伪标签与数据构建（W6）
- metrics: 多样性、hack 检测、profiling 解析
"""
