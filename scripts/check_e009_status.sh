#!/bin/bash
cd ~/projects/FPOT

squeue -u $USER -h -o "%i" | grep -oP '_\K[0-9]+' | sort -n -u > /tmp/running_idx.txt

ls slurm/logs/e009_nm_ablation_cube_double_*_*.out 2>/dev/null | while read f; do
    if tail -1 "$f" 2>/dev/null | grep -q "Done idx="; then
        basename "$f" .out | sed 's/.*_//'
    fi
done | sort -n -u > /tmp/done_idx.txt

> /tmp/crashed_idx.txt
for f in slurm/logs/e009_nm_ablation_cube_double_*_*.err; do
    if grep -q "Traceback\|Error executing\|RESOURCE_EXHAUSTED\|DUE TO PREEMPTION\|DUE TO TIME LIMIT" "$f" 2>/dev/null; then
        idx=$(basename "$f" .err | sed 's/.*_//')
        if ! grep -qx "$idx" /tmp/running_idx.txt && ! grep -qx "$idx" /tmp/done_idx.txt; then
            echo "$idx" >> /tmp/crashed_idx.txt
        fi
    fi
done
sort -n -u /tmp/crashed_idx.txt -o /tmp/crashed_idx.txt

> /tmp/missing_idx.txt
for i in $(seq 0 95); do
    if ! ls slurm/logs/e009_nm_ablation_cube_double_*_${i}.out >/dev/null 2>&1; then
        echo "$i" >> /tmp/missing_idx.txt
    fi
done

echo "=== SUMMARY ==="
echo "DONE:    $(wc -l < /tmp/done_idx.txt)"
echo "RUNNING: $(wc -l < /tmp/running_idx.txt)"
echo "CRASHED: $(wc -l < /tmp/crashed_idx.txt)"
echo "MISSING: $(wc -l < /tmp/missing_idx.txt)"
echo ""
echo "Crashed (resubmit these): $(cat /tmp/crashed_idx.txt | tr '\n' ',' | sed 's/,$//')"
echo "Missing (still queued):   $(cat /tmp/missing_idx.txt | tr '\n' ',' | sed 's/,$//')"
