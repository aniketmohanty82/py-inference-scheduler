"""Per-step decomposition of sampling time per trajectory into engine-path and tool time.

    python3 turn_decomp.py <verl log> [<verl log> ...]

For each `step:N - k:v - ...` line: turns/trajectory, generate_sequences mean
(time a trajectory spends waiting on the LLM path: hook + HTTP + engine),
tool_calls mean (sandbox commands), compute_score mean, each also per
assistant turn. Means over steps 2..N are printed last.
"""
import re
import sys

KEYS = {
    "turns": "num_turns/mean",
    "gen_traj": "timing_s/agent_loop/generate_sequences/mean",
    "tool_traj": "timing_s/agent_loop/tool_calls/mean",
    "score_traj": "timing_s/agent_loop/compute_score/mean",
    "gen_step": "timing_s/gen",
    "resp_len": "response_length/mean",
    "tool_max": "timing_s/agent_loop/tool_calls/max",
    "gen_max": "timing_s/agent_loop/generate_sequences/max",
}
NUM = r"(?:np\.float64\()?(-?[0-9.]+(?:e[-+]?[0-9]+)?)"

for path in sys.argv[1:]:
    rows = {}
    for ln in open(path, errors="replace"):
        m = re.search(r"step:(\d+) - ", ln)
        if not m:
            continue
        step = int(m.group(1))
        vals = {}
        for k, name in KEYS.items():
            mm = re.search(re.escape(name) + ":" + NUM, ln)
            if mm:
                vals[k] = float(mm.group(1))
        if "turns" in vals and "gen_traj" in vals:
            rows[step] = vals
    if not rows:
        print(f"== {path}: no step lines with agent_loop timings")
        continue
    print(f"== {path}")
    print("step  turns  LLM-path/traj  tool/traj  score/traj | LLM-path/turn  tool/turn | gen(step)  resp_len  LLM-path max  tool max")
    acc = {}
    for step in sorted(rows):
        v = rows[step]
        t = v["turns"]
        lt, tt = v["gen_traj"] / t, v.get("tool_traj", 0) / t
        print(f"{step:4d}  {t:5.1f}  {v['gen_traj']:13.1f}  {v.get('tool_traj', 0):9.1f}  {v.get('score_traj', 0):10.1f} | {lt:13.2f}  {tt:9.2f} | {v.get('gen_step', 0):9.0f}  {v.get('resp_len', 0):8.0f}  {v.get('gen_max', 0):12.0f}  {v.get('tool_max', 0):8.0f}")
        if step >= 2:
            for k, x in (("turns", t), ("lt", lt), ("tt", tt), ("gen_traj", v["gen_traj"]), ("tool_traj", v.get("tool_traj", 0))):
                acc.setdefault(k, []).append(x)
    if acc:
        n = len(acc["lt"])
        print(f"mean steps 2..{max(rows)} (n={n}): turns {sum(acc['turns'])/n:.1f}  LLM-path/traj {sum(acc['gen_traj'])/n:.1f} s  tool/traj {sum(acc['tool_traj'])/n:.1f} s  LLM-path/turn {sum(acc['lt'])/n:.2f} s  tool/turn {sum(acc['tt'])/n:.2f} s")
