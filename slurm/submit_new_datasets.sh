#!/usr/bin/env bash
# Fire 3 models × 6 datasets = 18 independent sbatch jobs for feature extraction.
# No --dependency chaining. Pending on GPU quota is expected and will auto-release.
#
# Env overrides:
#   MODELS_FILTER=qwen2.5-7b            only this tag (comma-sep allowed)
#   DATASETS_FILTER=belebele,gsm8k      only these datasets
#   DRY_RUN=1                            print what would be submitted

set -uo pipefail
cd /data/jehc223/NIPS2026
mkdir -p extraction/logs

MODELS=(
  "qwen2.5-7b:Qwen/Qwen2.5-7B-Instruct"
  "llama3.1-8b:meta-llama/Llama-3.1-8B-Instruct"
  "mistral-7b-v0.3:mistralai/Mistral-7B-Instruct-v0.3"
)
DATASETS=(
  # 6 new QA datasets (already extracted at the new setting; left here so a
  # future full re-run is one DATASETS_FILTER away)
  gsm8k math commonsenseqa theoremqa mmlu belebele
  # 7 old datasets, re-extracted at the new setting (chat template + 1024
  # max_new_tokens + Final-answer instructions). retrievalqa is intentionally
  # absent — its legacy raw features remain in B2 under features_legacy_512_no_chat.
  geometry_of_truth_cities easy2hard_amc metatool_task1
  common_claim_3class when2call_3class fava ragtruth
)

MODELS_FILTER=${MODELS_FILTER:-}
DATASETS_FILTER=${DATASETS_FILTER:-}
DRY_RUN=${DRY_RUN:-}

in_filter() {  # $1=value, $2=csv-filter (empty = match all)
  local val=$1 filt=$2
  [[ -z "$filt" ]] && return 0
  IFS=',' read -ra parts <<< "$filt"
  for p in "${parts[@]}"; do [[ "$val" == "$p" ]] && return 0; done
  return 1
}

submitted=0
for m in "${MODELS[@]}"; do
  tag=${m%%:*}; id=${m#*:}
  in_filter "$tag" "$MODELS_FILTER" || continue
  for ds in "${DATASETS[@]}"; do
    in_filter "$ds" "$DATASETS_FILTER" || continue
    cmd=(sbatch
      --job-name=ex_${tag}_${ds}
      --output=extraction/logs/ex_${tag}_${ds}_%j.out
      --error=extraction/logs/ex_${tag}_${ds}_%j.err
      slurm/30_extract.sbatch "$tag" "$id" "$ds")
    if [[ -n "$DRY_RUN" ]]; then
      echo "DRY: ${cmd[*]}"
    else
      "${cmd[@]}"
    fi
    submitted=$((submitted + 1))
  done
done

echo ""
echo "Submitted $submitted job(s)."
[[ -z "$DRY_RUN" ]] && echo "Monitor: squeue -u \$USER -o '%.9i %.12P %.25j %.8T %.10M %R'"
