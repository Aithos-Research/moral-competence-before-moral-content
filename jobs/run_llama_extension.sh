#!/usr/bin/env bash
# Extend the fintech + smart_home 8-model sweeps with Llama (dedicated deployment).
# Launches 4 jobs in parallel:
#   fintech ambig (100ep) + fintech neut (20ep) + smart_home ambig (100ep) + smart_home neut (20ep)
# All four land in dirs matching the existing __{scenario}_full__{phase} pattern
# so the dashboard renderers pick them up alongside the other 8 models.
set -u

MODEL="fireworks/accounts/aithos-sicco/deployments/hc3sp18t"
TAG="llama"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE=/c/tmp/win_certs.pem
export REQUESTS_CA_BUNDLE=/c/tmp/win_certs.pem

declare -a JOBS=(
  "18_fintech_transparency:ambiguous:100:fintech_full"
  "18_fintech_transparency:neutered:20:fintech_full"
  "23_smart_home_upsell:ambiguous:100:smart_home_full"
  "23_smart_home_upsell:neutered:20:smart_home_full"
)

PIDS=()
for j in "${JOBS[@]}"; do
  IFS=":" read -r SCEN FILTER EPOCHS SUFFIX <<< "$j"
  LOG_DIR="logs/${SCEN}__${TAG}__${SUFFIX}__${FILTER}"
  mkdir -p "$LOG_DIR"
  PYTHONIOENCODING=utf-8 python -m src.perturbations.run_inspect_cube \
    --scenario "$SCEN" \
    --model "$MODEL" \
    --dominance-filter "$FILTER" \
    --epochs "$EPOCHS" \
    --max-connections 10 \
    --turns 6 \
    --log-dir "$LOG_DIR" \
    > "$LOG_DIR/stdout.log" 2> "$LOG_DIR/stderr.log" &
  PIDS+=($!)
  echo "launched ${SCEN}/${FILTER} (PID ${PIDS[-1]}) -> $LOG_DIR"
done

F=0
for p in "${PIDS[@]}"; do wait "$p" || F=$((F+1)); done
echo "llama extension done (failures=$F)"
exit "$F"
