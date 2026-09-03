"""Paired comparison of the capped 128-task arms, from recorded rollout lines."""

import gzip
import re
import statistics as st

TMP = "/usr/local/google/home/aniketmohanty/.claude/jobs/68141fe0/tmp"
PAT = (
    r"in (\d+)s \(setup=(\d+)s agentflow=(\d+)s \[llm=(\d+)s/(\d+) steps\] "
    r"evaluator=(\d+)s"
)


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="ignore") as f:
        return [tuple(map(int, r)) for r in {m for m in re.findall(PAT, f.read())}]


def q(v, p):
    return sorted(v)[min(len(v) - 1, int(p / 100 * len(v)))]


arms = {
    "store": load(f"{TMP}/pair2_store_driver.log"),
    "recompute": load(f"{TMP}/pair2_recompute_driver.log"),
}
out = {}
for name, rows in arms.items():
    tot, setup, flow, llm, steps, ev = (list(c) for c in zip(*rows))
    lpt = [l / s for l, s in zip(llm, steps) if s]
    tool = [(f - l) / s for f, l, s in zip(flow, llm, steps) if s]
    out[name] = dict(
        n=len(rows), turns_mean=st.mean(steps), turns_med=st.median(steps),
        turns_total=sum(steps), capped=sum(1 for s in steps if s >= 25),
        llm_total=sum(llm), llm_per_turn=st.median(lpt), llm_per_turn_mean=st.mean(lpt),
        tool_per_turn=st.median(tool), agentflow_med=st.median(flow),
        setup_med=st.median(setup), setup_p95=q(setup, 95),
        traj_med=st.median(tot), traj_p95=q(tot, 95),
    )

s, r = out["store"], out["recompute"]
print(f"{'metric':<26}{'store':>12}{'recompute':>12}{'delta':>10}")
for k, label, unit in [
    ("n", "trajectories", ""), ("turns_mean", "turns mean", ""),
    ("turns_med", "turns median", ""), ("turns_total", "turns total", ""),
    ("capped", "hit 25-turn cap", ""), ("llm_total", "llm seconds total", "s"),
    ("llm_per_turn", "llm/turn median", "s"), ("llm_per_turn_mean", "llm/turn mean", "s"),
    ("tool_per_turn", "tool/turn median", "s"), ("agentflow_med", "agentflow median", "s"),
    ("setup_med", "setup median", "s"), ("setup_p95", "setup p95", "s"),
    ("traj_med", "trajectory median", "s"), ("traj_p95", "trajectory p95", "s"),
]:
    a, b = s[k], r[k]
    d = f"{(a - b) / b * 100:+.1f}%" if b else ""
    print(f"{label:<26}{a:>12.1f}{b:>12.1f}{d:>10}")
