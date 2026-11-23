#!/bin/bash
# Print the last throughput / memory metrics from each experiment's latest job, and check
# that NCCL used the libfabric (Slingshot RDMA) plugin. Without it NCCL falls back to TCP
# sockets (~70x slower) and the comparison is meaningless.
# Usage: ./batch/7b/scaling/summarize_experiments.sh   (run from the repo root)
#
# Compare after the first ~30 steps; early steps include torch.compile warm-up.
# TPS is per device; with TP/PP/CP there are fewer data-parallel ranks, so compare
# total tokens/s (TPS x 16 GPUs) across variants rather than per-device TPS alone.

printf "%-10s %12s %10s %10s  %-6s %s\n" variant "TPS/device" "active GiB" "resvd GiB" "RDMA" log
for d in logs/exp-*/; do
  v=$(basename "$d"); v=${v#exp-}
  out=$(ls -t "$d"exp_7b-*.out 2>/dev/null | head -1)
  [ -n "$out" ] || continue
  err=${out%.out}.err
  logs=("$out"); [ -f "$err" ] && logs+=("$err")   # metrics may be on stdout or stderr
  last() { grep -hF "$1" "${logs[@]}" 2>/dev/null | tail -1 | sed 's/.*=//'; }
  tps=$(last "throughput/device/TPS (actual avg)")
  act=$(last "GPU active mem (GiB)")
  res=$(last "GPU reserved mem (GiB)")

  # NCCL INFO logs are written per rank to logs/<run>/nccl-*.log
  if ls "$d"nccl-*.log >/dev/null 2>&1; then
    if grep -qi "using network.*libfabric" "$d"nccl-*.log 2>/dev/null; then rdma=yes
    elif grep -qi "using network" "$d"nccl-*.log 2>/dev/null; then rdma="NO!"
    else rdma="?"; fi
  else rdma="?"; fi

  if [ -z "$tps" ]; then
    msg=$(grep -hE "Error|error" "${logs[@]}" 2>/dev/null | tail -1)
    printf "%-10s %12s %10s %10s  %-6s %s\n" "$v" "-" "-" "-" "$rdma" "no metrics / failed: ${msg:0:70}"
  else
    printf "%-10s %12s %10s %10s  %-6s %s\n" "$v" "$tps" "${act:--}" "${res:--}" "$rdma" "$out"
  fi
done
