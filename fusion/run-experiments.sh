#!/usr/bin/env bash
# Usage: bash fusion/run-experiments.sh <model> [setting]
#   setting: "old" (default, Type-1 dataset GT) or "new" (Type-2 model-correctness)
# Runs all 8 experiments sequentially with checkpoint/resume where supported.
# Order matters: exp1b uses raw features, so it must run while extraction features for $MODEL exist.

set -u
MODEL="${1:-qwen2.5-7b}"
SETTING="${2:-old}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY_BIN:-python}"
if [ "$SETTING" = "new" ]; then
    RESULTS_BASE="$REPO_ROOT/fusion/results_correctness"
else
    RESULTS_BASE="$REPO_ROOT/fusion/results"
fi
LOG_DIR="$RESULTS_BASE/$MODEL/logs"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

run() {
    local name=$1 ; shift
    local logfile="$LOG_DIR/$name.log"
    echo "=== [$(date +%H:%M:%S)] $name on $MODEL (setting=$SETTING) ==="
    if "$@" --setting "$SETTING" > "$logfile" 2>&1; then
        echo "    OK → $logfile"
    else
        echo "    FAIL → $logfile (exit $?)"
        tail -20 "$logfile"
        return 1
    fi
}

# Fast experiments first (minutes each)
run exp1  $PY -u fusion/exp1_oracle_complete.py   --model $MODEL
# exp1b: in new setting the oracle baseline is already ≈1.0 (per-example oracle
# saturates with 12 baselines on binary correctness), so adding 11 raw views
# yields zero headroom and costs ~24h per model. Skip exp1b for new setting.
if [ "$SETTING" != "new" ]; then
    run exp1b $PY -u fusion/exp1b_oracle_with_raw.py  --model $MODEL
fi
run exp5  $PY -u fusion/exp5_probe_clustering.py  --model $MODEL
run exp6  $PY -u fusion/exp6_fava_extension.py    --model $MODEL
run v21   $PY -u fusion/baseline_only_v21_winning.py --model $MODEL

# Slow experiments last (~1–3h each)
run exp2  $PY -u fusion/exp2_probe_ladder.py      --model $MODEL
run exp3  $PY -u fusion/exp3_leave_one_out.py     --model $MODEL
run exp4  $PY -u fusion/exp4_pipeline_ablation.py --model $MODEL

echo
echo "=== Done: $MODEL ($SETTING) ==="
ls -la "$RESULTS_BASE/$MODEL"/*.json 2>/dev/null
