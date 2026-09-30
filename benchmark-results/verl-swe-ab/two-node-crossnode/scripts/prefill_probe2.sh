#!/bin/bash
# Uncached vs cached prefill cost on one live engine, for the latency ladder (v2).
#
# v1 reused the same deterministic prompts on every attempt and let each
# longer prompt extend the previous one, so only the first attempt's shortest
# prompt was truly cold. v2 salts every prompt with a per-run nonce and a
# per-length seed (nothing shared beyond the first block), keeps every length
# under the engine's 32,768-token limit, and adds the per-turn case an agent
# loop actually pays: a cached context plus ~500 new observation tokens.
# Run while the engines are alive but idle, e.g. a rollout tail.
#   ./prefill_probe2.sh <kube ctx> <worker pod> <engine port> [model]
set -u
CTX=$1; POD=$2; PORT=$3; MODEL=${4:-Qwen/Qwen2.5-32B-Instruct}
kubectl --context "$CTX" exec -i "$POD" -c ray-worker -- python3 -u - "$PORT" "$MODEL" <<'PY'
import json, os, random, sys, time, urllib.error, urllib.request
port, model = sys.argv[1], sys.argv[2]
host = os.popen("hostname -i").read().split()[0]
base = f"http://{host}:{port}"
nonce = f"run{int(time.time())}"
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa "
         "quebec romeo sierra tango uniform victor whiskey xray yankee zulu apple river stone cloud green "
         "table light water mountain paper window garden silver copper bridge forest candle mirror").split()

def post(path, body, timeout=600):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = r.read()
    return time.perf_counter() - t, json.loads(out)

def complete(prompt):
    try:
        return post("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0})
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return post("/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1, "temperature": 0})
        print(f"HTTP {e.code}: {e.read()[:200]!r}", flush=True)
        return None, None

def words(seed, n):
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n))

def gpus_busy():
    # The trainer shares these GPUs; a probe that overlaps the update phase
    # measures contention, not prefill (seen 23:54 UTC: 2x slower). Sleep
    # first so the probe's own last prefill has left nvidia-smi's ~1 s
    # utilisation window, or the guard trips on itself (seen 00:18 UTC).
    time.sleep(1.5)
    util = [int(x) for x in os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read().split()]
    return max(util or [100]) > 5

print(f"engine {base} model {model} nonce {nonce}", flush=True)
complete(f"{nonce} warmup")  # connection + tokenizer warm-up, off the clock
for n in (1500, 3000, 6000, 12000, 20000):
    if gpus_busy():
        print(f"gpus busy before L{n}: stopping so the numbers stay uncontended", flush=True)
        sys.exit(3)
    cold = []
    for rep in range(2):
        prompt = f"{nonce} L{n} R{rep} " + words(f"{nonce}-{n}-{rep}", n)
        dt, out = complete(prompt)
        if out is None:
            break
        ptoks = out["usage"]["prompt_tokens"]
        cold.append((ptoks, dt))
    if not cold:
        continue
    warm_dt, _ = complete(prompt)  # exact repeat of the last cold prompt: prefix hit on everything but the tail block
    turn_dt, turn_out = complete(prompt + " " + words(f"{nonce}-{n}-turn", 400))  # cached context + ~500 new tokens
    new = turn_out["usage"]["prompt_tokens"] - ptoks
    cold_s = " ".join(f"{dt:6.3f}s({p / dt:6.0f} tok/s)" for p, dt in cold)
    print(f"prompt_tokens={ptoks:6d} cold x2: {cold_s}  warm_repeat={warm_dt:6.3f}s  turn(+{new} new tok)={turn_dt:6.3f}s", flush=True)
PY
