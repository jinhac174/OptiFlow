#!/bin/bash
# Usage: bash scripts/check_sweep_status.sh <sweep_name> <total_jobs>
# Example: bash scripts/check_sweep_status.sh e013a_nm_ablation_antmaze_remote 72
SWEEP=${1:?"Usage: $0 <sweep_name> <total_jobs>"}
TOTAL=${2:?"Usage: $0 <sweep_name> <total_jobs>"}
cd ~/projects/FPOT

squeue -u $USER -h -o "%j %i" | awk -v s="$SWEEP" '$1 == s {split($2, a, "_"); print a[2]}' | sort -n -u > /tmp/running_idx.txt

ls slurm/logs/${SWEEP}_*_*.out 2>/dev/null | while read f; do
    if tail -1 "$f" 2>/dev/null | grep -q "Done idx="; then
        basename "$f" .out | sed 's/.*_//'
    fi
done | sort -n -u > /tmp/done_idx.txt

> /tmp/crashed_idx.txt
for f in slurm/logs/${SWEEP}_*_*.err; do
    if grep -q "Traceback\|Error executing\|RESOURCE_EXHAUSTED\|DUE TO PREEMPTION\|DUE TO TIME LIMIT" "$f" 2>/dev/null; then
        idx=$(basename "$f" .err | sed 's/.*_//')
        if ! grep -qx "$idx" /tmp/running_idx.txt && ! grep -qx "$idx" /tmp/done_idx.txt; then
            echo "$idx" >> /tmp/crashed_idx.txt
        fi
    fi
done
sort -n -u /tmp/crashed_idx.txt -o /tmp/crashed_idx.txt

> /tmp/missing_idx.txt
for i in $(seq 0 $((TOTAL - 1))); do
    if ! ls slurm/logs/${SWEEP}_*_${i}.out >/dev/null 2>&1; then
        echo "$i" >> /tmp/missing_idx.txt
    fi
done

echo "=== $SWEEP SUMMARY ==="
echo "DONE:    $(wc -l < /tmp/done_idx.txt)"
echo "RUNNING: $(wc -l < /tmp/running_idx.txt)"
echo "CRASHED: $(wc -l < /tmp/crashed_idx.txt)"
echo "MISSING: $(wc -l < /tmp/missing_idx.txt)"
echo "Crashed: $(cat /tmp/crashed_idx.txt | tr '\n' ',' | sed 's/,$//')"
echo "Missing: $(cat /tmp/missing_idx.txt | tr '\n' ',' | sed 's/,$//')"
