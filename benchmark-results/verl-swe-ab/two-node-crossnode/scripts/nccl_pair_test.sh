#!/bin/bash
# Two-pod NCCL check: one process per worker pod, all_reduce over the cross-node
# NCCL transport, with NCCL_DEBUG=INFO so the chosen NET (IB vs Socket), the
# GID index and any timeout are visible.  Usage:
#   ./nccl_pair_test.sh <kube ctx> <pod1> <pod2> [extra NCCL env, e.g. NCCL_IB_GID_INDEX=3]
set -u
CTX=$1; P1=$2; P2=$3; shift 3; EXTRA="$*"
k(){ kubectl --context "$CTX" "$@"; }
MASTER=$(k exec "$P1" -c ray-worker -- hostname -i | tr -d ' \n')
echo "master=$MASTER extra='$EXTRA'"
cat > /tmp/nccl_pair_test.py <<'EOF'
import os, time, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); torch.cuda.set_device(0)
t0 = time.time()
dist.init_process_group("nccl", rank=rank, world_size=2, timeout=__import__("datetime").timedelta(seconds=120))
x = torch.ones(64 * 1024 * 1024, device="cuda")  # 256 MB
dist.all_reduce(x); torch.cuda.synchronize()
t1 = time.time()
for _ in range(5): dist.all_reduce(x)
torch.cuda.synchronize(); t2 = time.time()
print(f"RANK{rank}: init+first all_reduce {t1 - t0:.1f}s, 5x256MB all_reduce {t2 - t1:.2f}s -> {5 * 2 * 256 / 1024 / max(1e-6, t2 - t1):.1f} GB/s busbw-ish", flush=True)
dist.destroy_process_group()
EOF
for P in "$P1" "$P2"; do k cp /tmp/nccl_pair_test.py "$P:/tmp/nccl_pair_test.py" -c ray-worker >/dev/null; done
run(){ k exec "$1" -c ray-worker -- bash -c "cd /tmp && env $EXTRA NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_SOCKET_IFNAME=eth0 MASTER_ADDR=$MASTER MASTER_PORT=${PORT:-29777} WORLD_SIZE=2 RANK=$2 CUDA_VISIBLE_DEVICES=${GPU:-7} timeout 170 python3 /tmp/nccl_pair_test.py > /tmp/nccl_full_$2.log 2>&1; echo exit=\$?; grep -E 'RANK|Using network|Plugin|Connected|Init COMPLETE|WARN|Error|error|Timeout|timed out|abort' /tmp/nccl_full_$2.log | tail -n 20; echo ---last---; tail -n 3 /tmp/nccl_full_$2.log"; }
run "$P2" 1 > /tmp/nccl_r1_${PORT:-29777}.out 2>&1 &
run "$P1" 0 > /tmp/nccl_r0_${PORT:-29777}.out 2>&1
wait
echo "--- rank0 ($P1) ---"; cut -c1-200 /tmp/nccl_r0_${PORT:-29777}.out; echo "--- rank1 ($P2) ---"; cut -c1-200 /tmp/nccl_r1_${PORT:-29777}.out
