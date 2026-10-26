"""严格 verifier 的冻结守卫（W5 人工决策门）。

P7 审计报告（`reports/P7_reward_hacking_audit.md`）通过人工审阅后，W3 的严格 verifier 被冻结为
`verifier-v1.0`（annotated tag，指向 `467b015`——verifier 最后一次改动的 commit）。
W3/W4/W5 的全部结论都建立在这一份实现上：训练奖励与评测共用它，`strict_acc` 由它给出，
因此 reward gap、hack rate、各防御的代价都以它为基准。

改动 verifier 就等于让那些数字失去可比性，所以这里把文件内容的 sha256 钉死：改一个字节就红灯。

**红灯之后要做什么**（不要只是更新下面的常量）：

1. 在 `docs/troubleshooting.md` 里记录改了什么、为什么；
2. 评估受影响的实验（至少 `EXP-W03-*`、`EXP-W04-*`、`EXP-W05-*`），需要重评的用
   `eval/run_eval.py --rescore-only` 重判，并更新 `reports/results_master.md`；
3. 冻结新版本（`verifier-v1.1` …），再把下面的常量和 `VERIFIER_VERSION` 一起更新。

sha256 而不是 git blob id：守卫要在没有 git 的环境（远端训练机的 tarball、CI 容器）里也成立。

**行尾归一化不是可选项**：本仓库在 Windows 侧 `core.autocrlf=true`，同一个 commit 在本地检出成
CRLF、在远端 Linux 训练机上是 LF，裸字节 sha256 两边必然不同（实测相差
`ac1117f…` vs `cba5e9b…`），守卫会在远端误报。因此这里统一把 CRLF/CR 归一成 LF 再算 sha256，
钉死的值就是仓库实际存储的内容的哈希，跨平台一致。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_DIR = REPO_ROOT / "src" / "selfevolve" / "verifier"

#: 判分器版本串，与 `reports/P7_reward_hacking_audit.md` 和各 eval summary 的记录一致。
VERIFIER_VERSION = "selfevolve-strict-verifier-1.0"

#: `verifier-v1.0`（commit 467b015）冻结时各文件**行尾归一化后**的 sha256。
FROZEN_SHA256 = {
    "__init__.py": "471afe90bf13b536df748aaa2934d8f778e1087dc724bfecc5230c3aa109984f",
    "math_verifier.py": "ac1117f2fecabdf671aa61d714b46953cfdfc4a2719e45bf7a0b41f4114842ee",
}


def _sha256(path: Path) -> str:
    """行尾归一成 LF 后再算 sha256（见模块 docstring：autocrlf 会让裸字节跨平台不一致）。"""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("filename", sorted(FROZEN_SHA256))
def test_verifier_file_is_frozen(filename: str) -> None:
    path = VERIFIER_DIR / filename
    assert path.is_file(), f"{filename} 不存在：verifier 被移动或删除了"
    actual = _sha256(path)
    assert actual == FROZEN_SHA256[filename], (
        f"{filename} 与 {VERIFIER_VERSION}（tag verifier-v1.0 / commit 467b015）不一致：\n"
        f"  冻结值 {FROZEN_SHA256[filename]}\n"
        f"  当前值 {actual}\n"
        "判分器已冻结。改动会让 W3/W4/W5 的 strict_acc、reward gap、hack rate 失去可比性——"
        "先按本文件 docstring 的三步走（记录变更、重评受影响实验、冻结新版本），"
        "不要只更新这里的常量。"
    )


def test_no_unfrozen_files_in_verifier_package() -> None:
    """新增文件也要走冻结流程，否则严格判定可以从一个没被守卫的模块里被改掉。"""
    present = {
        p.name
        for p in VERIFIER_DIR.iterdir()
        if p.is_file() and p.suffix == ".py"
    }
    unfrozen = present - set(FROZEN_SHA256)
    assert not unfrozen, (
        f"verifier 包里有未冻结的文件：{sorted(unfrozen)}。"
        "新增模块请一并登记到 FROZEN_SHA256，并按 docstring 评估受影响的实验。"
    )
