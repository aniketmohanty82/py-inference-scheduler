#!/bin/bash
# Pod-side router boot check. Launched DETACHED (never as a kubectl exec
# foreground command — an exec whose command backgrounds a daemon can hang
# the exec stream indefinitely). Writes PASS/FAIL to /root/boot_check.result.
rm -f /root/boot_check.result
pkill -x scheduler 2>/dev/null; sleep 2
INODE=$(awk "\$2 ~ /:1F40\$/ {print \$10}" /proc/net/tcp | head -1)
if [ -n "$INODE" ]; then
  for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "socket:\[$INODE\]" && kill -9 ${p#/proc/} 2>/dev/null; done
  sleep 1
fi
cd /root/pis && nohup setsid python -m integration.slime --host 0.0.0.0 --port 8000 \
  --config /root/prof_champion.yaml > /root/router_boot_check.log 2>&1 < /dev/null &
sleep 8
W=$(curl -s -m 3 localhost:8000/workers)
pkill -x scheduler 2>/dev/null
if [ "$W" = '{"workers":[]}' ]; then echo PASS > /root/boot_check.result
else echo "FAIL: workers=$W" > /root/boot_check.result; fi
