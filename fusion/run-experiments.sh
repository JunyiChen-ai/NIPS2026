#!/usr/bin/env bash
# Usage: bash fusion/run_all_experiments.sh <model>
# Runs all 8 experiments sequentially with checkpoint/resume where supported.
# Order matters: exp1b uses raw features, so it must run while extraction/features/$MODEL exists.

set -u
MODEL="${1:-qwen2.5-7b}"
PY=/home/junyi/miniconda3/envs/WWW/bin/python
LOG_DIR=/home/junyi/NIPS2026/fusion/results/$MODEL/logs
mkdir -p "$LOG_DIR"

cd /home/junyi/NIPS2026

run() {
    local name=$1 ; shift
    local logfile="$LOG_DIR/$name.log"
    echo "=== [$(date +%H:%M:%S)] $name on $MODEL ==="
    if "$@" > "$logfile" 2>&1; then
        echo "    OK → $logfile"
    else
        echo "    FAIL → $logfile (exit $?)"
        tail -20 "$logfile"
        return 1
    fi
}

# Fast experiments first (minutes each)
run exp1  $PY -u fusion/exp1_oracle_complete.py   --model $MODEL
run exp1b $PY -u fusion/exp1b_oracle_with_raw.py  --model $MODEL
run exp5  $PY -u fusion/exp5_probe_clustering.py  --model $MODEL
run exp6  $PY -u fusion/exp6_fava_extension.py    --model $MODEL
run v21   $PY -u fusion/baseline_only_v21_winning.py --model $MODEL

# Slow experiments last (~1–3h each)
run exp2  $PY -u fusion/exp2_probe_ladder.py      --model $MODEL
run exp3  $PY -u fusion/exp3_leave_one_out.py     --model $MODEL
run exp4  $PY -u fusion/exp4_pipeline_ablation.py --model $MODEL

echo
echo "=== Done: $MODEL ==="
ls -la /home/junyi/NIPS2026/fusion/results/$MODEL/*.json
