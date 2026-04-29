#!/usr/bin/env bash
# Emits one-line events on SLURM job state transitions for our extraction jobs.
# Event format: [HH:MM:SS] <jobid> <name> <prev> -> <curr> | <health>
#
# Health signals (at transitions and on leave-queue):
#   RUNNING entry  : scans stdout/stderr for model-load / batch-progress / error markers
#   LEFT-QUEUE     : checks sentinel + meta.json (gen_texts count, gold_answers presence),
#                    or sacct exit code if meta missing, or "uploaded+deleted" if watcher ran
#
# Only state transitions emit events (not every poll tick), so the stream stays quiet
# unless something is happening. Failure signatures (OOM, Traceback, FAILED sacct) always emit.

set -uo pipefail
cd /data/jehc223/NIPS2026

POLL=${POLL:-30}
STATE_FILE=/tmp/.monitor_jobs_state.$USER

save_state() {  # flat: "id=STATE name" per line
  : > "$STATE_FILE.new"
  for id in "${!prev_state[@]}"; do
    printf '%s=%s %s\n' "$id" "${prev_state[$id]}" "${known_name[$id]:-unknown}" >> "$STATE_FILE.new"
  done
  mv "$STATE_FILE.new" "$STATE_FILE"
}

load_state() {
  [[ -f "$STATE_FILE" ]] || return 0
  while read -r line; do
    id_state=${line%% *}
    name=${line#* }
    id=${id_state%%=*}
    st=${id_state#*=}
    prev_state[$id]=$st
    known_name[$id]=$name
  done < "$STATE_FILE"
}

# Parse job-name → model_tag, dataset (best-effort; smoke job is special-cased).
# Job names: ex_<TAG>_<DATASET>. TAG is one of the 3 known model tags; DATASET
# can contain underscores (e.g. geometry_of_truth_cities, common_claim_3class),
# so we match by known TAG prefix instead of heuristic underscore splitting.
parse_name() {
  local name=$1
  local body=${name#ex_}
  if [[ "$body" == smoke_* ]]; then
    echo "qwen2.5-7b belebele"; return
  fi
  for tag in qwen2.5-7b llama3.1-8b mistral-7b-v0.3; do
    if [[ "$body" == "${tag}_"* ]]; then
      echo "$tag ${body#${tag}_}"; return
    fi
  done
  # Fallback: original heuristic (last underscore split). Only hits if a new
  # model tag is introduced without updating the list above.
  echo "${body%_*} ${body##*_}"
}

health_running() {
  local jobid=$1
  local glob="extraction/logs/ex_*_${jobid}.out"
  local stdout; stdout=$(compgen -G "$glob" 2>/dev/null | head -1)
  local stderr=${stdout%.out}.err
  [[ -z "$stdout" ]] && { echo "no-log-yet"; return; }
  if [[ -s "$stderr" ]] && grep -qE "Traceback|OutOfMemoryError|CUDA error|ImportError|FATAL" "$stderr" 2>/dev/null; then
    echo "ERR:$(grep -E 'Traceback|OutOfMemoryError|CUDA error|ImportError|FATAL' "$stderr" | head -1 | tr -d '\n' | cut -c1-140)"
    return
  fi
  if grep -q "Processing " "$stdout" 2>/dev/null; then
    local last; last=$(grep -E 'Processing|[0-9]+it/s|/s]' "$stdout" | tail -1 | tr -d '\n' | cut -c1-100)
    echo "progressing: $last"
  elif grep -q "Loading model" "$stdout" 2>/dev/null; then
    echo "loading-model"
  else
    echo "starting"
  fi
}

health_left_queue() {
  local jobid=$1 name=$2
  read tag ds < <(parse_name "$name")
  local dir="extraction/features/${tag}/${ds}"
  local sentinel="${dir}/.extraction_done"

  if grep -qxF "${tag}/${ds}" slurm/.uploaded_features.log 2>/dev/null; then
    echo "uploaded+deleted ✓"
    return
  fi
  # Phase 1 datasets produce per-split subdirs (train/eval/val/test/...);
  # Phase 2 produces a single all/. Find any split subdir with meta.json.
  local any_meta; any_meta=$(find "$dir" -mindepth 2 -maxdepth 2 -name meta.json 2>/dev/null | head -1)
  if [[ -n "$any_meta" ]]; then
    # Strong validation (exit 0 = OK, 1 = FAIL with reason)
    local v; v=$(python slurm/validate_extraction.py "$tag" "$ds" 2>&1 | head -1 | cut -c1-200)
    echo "$v"
    return
  fi
  local sa; sa=$(sacct -j "$jobid" -o State,ExitCode -n -P 2>/dev/null | head -1)
  if [[ -n "$sa" ]]; then
    # also scan stderr briefly
    local glob="extraction/logs/ex_*_${jobid}.err"
    local stderr; stderr=$(compgen -G "$glob" 2>/dev/null | head -1)
    local tail_err=""
    if [[ -n "$stderr" && -s "$stderr" ]]; then
      tail_err=" err:$(tail -3 "$stderr" | tr '\n' ' ' | cut -c1-120)"
    fi
    echo "NO meta — sacct=$sa${tail_err}"
  else
    echo "NO meta, no sacct (may still be finishing)"
  fi
}

declare -A prev_state
declare -A known_name
load_state

while true; do
  declare -A curr_state=()
  while IFS='|' read -r id name state; do
    [[ -z "${id:-}" ]] && continue
    curr_state[$id]=$state
    known_name[$id]=$name
  done < <(squeue -u "$USER" -h -o '%i|%j|%T' 2>/dev/null || true)

  # transitions (includes new appearances: prev=NEW)
  for id in "${!curr_state[@]}"; do
    cs=${curr_state[$id]}
    ps=${prev_state[$id]:-NEW}
    if [[ "$ps" != "$cs" ]]; then
      name=${known_name[$id]}
      note=""
      if [[ "$cs" == "RUNNING" ]]; then
        sleep 4
        note=" | $(health_running "$id")"
      elif [[ "$cs" =~ ^(FAILED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|CANCELLED)$ ]]; then
        note=" | FAILED_STATE"
      fi
      printf '[%s] %s %s %s -> %s%s\n' "$(date +%H:%M:%S)" "$id" "$name" "$ps" "$cs" "$note"
    fi
    prev_state[$id]=$cs
  done

  # departures (left the queue between ticks)
  for id in "${!prev_state[@]}"; do
    if [[ -z "${curr_state[$id]:-}" ]]; then
      name=${known_name[$id]:-unknown}
      sleep 5  # let watcher / fs catch up
      note=$(health_left_queue "$id" "$name")
      printf '[%s] %s %s LEFT-QUEUE | %s\n' "$(date +%H:%M:%S)" "$id" "$name" "$note"
      unset "prev_state[$id]"
    fi
  done

  save_state
  sleep "$POLL"
done
