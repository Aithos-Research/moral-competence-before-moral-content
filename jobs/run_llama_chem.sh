#!/usr/bin/env bash
# Add Llama to the chem_spill cube (mirrors the chem_full_8 layout).
set -u
MODEL="fireworks/accounts/aithos-sicco/deployments/hc3sp18t"
TAG="llama"
SCEN="13_chemical_spill"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE=/c/tmp/win_certs.pem
export REQUESTS_CA_BUNDLE=/c/tmp/win_certs.pem
PIDS=()
for j in "ambiguous:100" "neutered:20"; do
  FILTER="${j%%:*}"; EPOCHS="${j##*:}"
  LOG_DIR="logs/${SCEN}__${TAG}__chem_full__${FILTER}"
  mkdir -p "$LOG_DIR"
  PYTHONIOENCODING=utf-8 python -m src.perturbations.run_inspect_cube \
    --scenario "$SCEN" --model "$MODEL" \
    --dominance-filter "$FILTER" --epochs "$EPOCHS" \
    --max-connections 10 --turns 6 --log-dir "$LOG_DIR" \
    > "$LOG_DIR/stdout.log" 2> "$LOG_DIR/stderr.log" &
  PIDS+=($!)
  echo "launched chem/$FILTER (PID ${PIDS[-1]}) -> $LOG_DIR"
done
F=0; for p in "${PIDS[@]}"; do wait "$p" || F=$((F+1)); done
echo "llama chem done (failures=$F)"; exit "$F"
