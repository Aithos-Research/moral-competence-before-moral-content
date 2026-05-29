#!/usr/bin/env bash
# Full fintech v7 cube — 8 models, 5 paraphrases. Mirrors run_chem_full_8.sh.
# Phase 1 (ambiguous):  100 epochs/cell × 25 cells = 2500 trials/model
# Phase 2 (neutered):    20 epochs/cell × 50 cells = 1000 trials/model
# Total: 3500 trials/model × 8 models = 28,000 trials.
#
# Assumes prompts/18_fintech_transparency/cells/ is the FIXED v7 build:
#   - violation_detection: content_match (>= 2 disclosure markers in text)
#   - YAML neutered_{toward,against} swapped to match polarity convention
set -u

SCENARIO=18_fintech_transparency

MODELS=(
  "anthropic/claude-sonnet-4-6:sonnet46"
  "openai/gpt-5.4:gpt54"
  "google/gemini-2.5-pro:gemini25pro"
  "mistral/mistral-large-2512:mistral2512"
  "fireworks/accounts/fireworks/models/glm-5p1:glm5p1"
  "fireworks/accounts/fireworks/models/deepseek-v4-pro:dsv4pro"
  "fireworks/accounts/fireworks/models/kimi-k2p6:kimik2p6"
  "fireworks/accounts/fireworks/models/qwen3p6-plus:qwen3p6plus"
)

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SSL_CERT_FILE=/c/tmp/win_certs.pem
export REQUESTS_CA_BUNDLE=/c/tmp/win_certs.pem

run_phase () {
  local FILTER="$1"; local EPOCHS="$2"; local PHASE="$3"
  echo "==== phase: $PHASE ($FILTER, $EPOCHS epochs) ===="
  local PIDS=()
  for spec in "${MODELS[@]}"; do
    local model="${spec%%:*}"; local tag="${spec##*:}"
    local log_dir="logs/${SCENARIO}__${tag}__fintech_full__${PHASE}"
    mkdir -p "$log_dir"
    PYTHONIOENCODING=utf-8 python -m src.perturbations.run_inspect_cube \
      --scenario "$SCENARIO" \
      --model "$model" \
      --dominance-filter "$FILTER" \
      --epochs "$EPOCHS" \
      --max-connections 10 \
      --turns 6 \
      --log-dir "$log_dir" \
      > "$log_dir/stdout.log" 2> "$log_dir/stderr.log" &
    PIDS+=($!)
    echo "  launched $tag (PID ${PIDS[-1]}) -> $log_dir"
  done
  local F=0
  for p in "${PIDS[@]}"; do wait "$p" || F=$((F+1)); done
  echo "==== $PHASE done (failures=$F) ===="
  return "$F"
}

run_phase ambiguous 100 ambiguous
F1=$?
run_phase neutered  20  neutered
F2=$?

echo "==== summary: ambiguous fail=$F1, neutered fail=$F2 ===="
exit $((F1 + F2))
