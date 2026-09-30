"""Cross-node KV sharing gate for the two-node fcstore arm.

Reads the harvest directory written by sched2.sh and checks, from three
independent instruments, that KV actually crossed nodes:

  1. hook MOVE lines: trajectories moved between engines on different hosts
  2. connector PULLSRC lines on each worker: keys pulled whose replica lives
     on the other host (cum_cross), i.e. KV written by the other node
  3. engine /metrics on each worker: prompt tokens served from
     external_kv_transfer on engines of BOTH nodes

and that flow control was live (FLOWCONTROL park/drop lines exist and the
run did not fall back to verl's balancer).

    python3 crossnode_gate.py <harvest dir>
"""

import glob
import os
import re
import sys

out = sys.argv[1]
fails: list[str] = []
def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        fails.append(msg)

drv = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", open(os.path.join(out, "fcstore.log"), errors="replace").read())

moves = re.findall(r"MOVE\[\d+\] rid=\S+ from=([\d.]+):\d+ to=([\d.]+):\d+ cross_node=(\d)", drv)
cross = sum(1 for _, _, c in moves if c == "1")
hosts = {h for m in moves for h in (m[0], m[1])}
check(len(moves) > 0, f"moves observed: {len(moves)} across hosts {sorted(hosts)}")
check(cross > 0, f"cross-node moves (holder on one host, next turn on the other): {cross}")

parks = len(re.findall(r"FLOWCONTROL park", drv)); drops = len(re.findall(r"FLOWCONTROL drop", drv))
check(parks + drops > 0, f"flow control live: parks {parks}, drops {drops}")
check(len(re.findall(r"falling back to verl", drv)) == 0, "no fallbacks to verl's balancer")

# PULLSRC lines from every source (driver log copy + worker Ray logs), last
# cumulative line per local host. Each TP rank keeps its own counters, so the
# per-host figure is the max over ranks, a lower bound on that host's pulls.
by_host: dict[str, tuple[int, int, int]] = {}
for path in glob.glob(os.path.join(out, "pullsrc_*.log")):
    for ln in open(path, errors="replace"):
        m = re.search(r"PULLSRC local=(\S+) .*cum_keys=(\d+) cum_cross=(\d+) cum_unknown=(\d+)", ln)
        if m:
            host, keys, xk, unk = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            if keys >= by_host.get(host, (0, 0, 0))[0]:
                by_host[host] = (keys, xk, unk)
check(len(by_host) >= 2, f"PULLSRC hosts reporting: {sorted(by_host)}")
for host, (keys, xk, unk) in sorted(by_host.items()):
    check(keys > 0, f"host {host}: {keys} keys pulled (max over ranks)")
    check(xk > 0, f"host {host}: {xk} keys pulled from the OTHER host ({100 * xk / max(1, keys):.0f}%), unknown {unk}")
if not by_host:
    check(False, "no PULLSRC lines with cumulative counts anywhere")

def ext_tokens(scrape: str) -> dict[str, float]:
    per_port: dict[str, float] = {}; port = None
    for ln in open(scrape, errors="replace"):
        if ln.startswith("=== PORT"):
            port = ln.split()[2]
        elif port and ln.startswith("vllm:prompt_tokens_by_source_total") and 'source="external_kv_transfer"' in ln:
            per_port[port] = float(ln.rsplit(" ", 1)[1])
    return per_port
for path in sorted(glob.glob(os.path.join(out, "fcstore_scrape_w*.log"))):
    ext = ext_tokens(path)
    served = {p: v for p, v in ext.items() if v > 0}
    check(len(served) > 0 and len(served) == len(ext),
          f"{os.path.basename(path)}: engines serving tier tokens {len(served)}/{len(ext)} ({sum(ext.values()):,.0f} tokens)")

master = open(os.path.join(out, "mooncake_master.log"), errors="replace").read() if os.path.exists(os.path.join(out, "mooncake_master.log")) else ""
seg_hosts = set(re.findall(r"(\d+\.\d+\.\d+\.\d+):\d+", master))
print(f"  INFO  mooncake master saw segment hosts: {sorted(seg_hosts)}")

print("\nCROSS-NODE GATE:", "PASS" if not fails else f"FAIL ({len(fails)})")
sys.exit(0 if not fails else 1)
