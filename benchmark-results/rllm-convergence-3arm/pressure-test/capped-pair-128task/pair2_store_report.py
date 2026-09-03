"""Sampling-phase record for the regime-2 capped store arm (training never finished)."""

import re
import statistics as st

LOG = "/usr/local/google/home/aniketmohanty/.claude/jobs/68141fe0/tmp/pair2_store_driver.log"
PAT = r"in (\d+)s \(setup=(\d+)s agentflow=(\d+)s \[llm=(\d+)s/(\d+) steps\] evaluator=(\d+)s"

txt = open(LOG, errors="ignore").read()
rows = [tuple(map(int, r)) for r in {m for m in re.findall(PAT, txt)}]
tot, setup, flow, llm, steps, ev = (list(c) for c in zip(*rows))
tool = [f - l for f, l in zip(flow, llm)]
lpt = [l / s for l, s in zip(llm, steps) if s]
tpt = [t / s for t, s in zip(tool, steps) if s]
n = len(rows)


def q(v, p):
    return sorted(v)[min(len(v) - 1, int(p / 100 * len(v)))]


solved = len(re.findall(r"mini-swe-agent: 1\.0", txt))
print(f"trajectories delivered : {n}   solved: {solved} ({solved / n * 100:.1f}%)")
print(f"turns                  : median {st.median(steps):.0f}  p95 {q(steps, 95)}  max {max(steps)}  total {sum(steps)}")
print(f"llm per turn           : median {st.median(lpt):.1f}s  mean {st.mean(lpt):.1f}s  p95 {q(lpt, 95):.1f}s")
print(f"tool per turn          : median {st.median(tpt):.1f}s")
print(f"trajectory wall        : median {st.median(tot):.0f}s  p95 {q(tot, 95)}s  max {max(tot)}s")
print(f"agentflow              : median {st.median(flow):.0f}s  max {max(flow)}s")
print(f"setup                  : median {st.median(setup):.0f}s  p95 {q(setup, 95)}s")
print(f"evaluator              : median {st.median(ev):.0f}s")
print(f"turn-cap hits (>=25)   : {sum(1 for s in steps if s >= 25)}")
print(f"timeout hits (>=10700s): {sum(1 for f in flow if f >= 10700)}")
counters = re.findall(r"consumed=(\d+), filtered=(\d+)", txt)
if counters:
    print(f"groups                 : consumed {counters[-1][0]}  filtered {counters[-1][1]}")
for pat, label in ((r"Attempt \d/3 failed", "sandbox retries"), (r"EnrichMismatchError", "enrich mismatches")):
    print(f"{label:<23}: {len(re.findall(pat, txt))}")
