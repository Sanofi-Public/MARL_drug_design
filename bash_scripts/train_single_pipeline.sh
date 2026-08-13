#!/bin/bash
set -e

LOGDIR="single_agent_pipeline_logs"
mkdir -p "$LOGDIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# Configuration
# ============================================================

# Single-agent config directory
CONFIG_DIR="configs/marl_configs/single_agent_props"

# Multi-agent (combined) config: FXa target + 3 RDKit properties
COMBINED_CONFIG="configs/marl_configs/fxa_rdkit/combined_w_pretrained_fxa_logp_molweight_tpsa.json"

# Output path for the combined pretrained model
# Must match `algorithm.pre_train_path` in $COMBINED_CONFIG
COMBINED_MODEL_DIR="ia2c_models/combined_mpo_rdkit"

# Property-to-config mapping
# Multi-agent properties (from $COMBINED_CONFIG): molweight, tpsa, logp
declare -a SINGLE_CONFIGS=(
    "${CONFIG_DIR}/single_agent_molweight_strict.json"
    "${CONFIG_DIR}/single_agent_tpsa_strict.json"
    "${CONFIG_DIR}/single_agent_logp_strict.json"
)

# ============================================================
# Phase 1: Train single-agent specialists
# ============================================================

TOTAL=${#SINGLE_CONFIGS[@]}

echo "=========================================="
echo "PHASE 1: Training Single-Agent Specialists"
echo "Pipeline started at $(date)"
echo "Found $TOTAL single-agent configs to train"
echo "=========================================="

CURRENT=0
FAILED=0
declare -a MODEL_PATHS=()

for config in "${SINGLE_CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$config" .json)

    echo ""
    echo "[$CURRENT/$TOTAL] Training $BASENAME..."
    echo "  Config: $config"

    if python train.py --config "$config" \
        2>&1 | tee "$LOGDIR/train_${BASENAME}_${TIMESTAMP}.txt"; then
        echo "[$CURRENT/$TOTAL] $BASENAME completed at $(date)"
    else
        echo "[$CURRENT/$TOTAL] WARNING: $BASENAME FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Phase 1 finished at $(date)"
echo "Trained: $((TOTAL - FAILED))/$TOTAL succeeded"
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED training run(s) failed!"
    echo "Aborting pipeline."
    exit 1
fi
echo "=========================================="

# ============================================================
# Phase 2: Combine single agents into multi-agent model
# ============================================================

echo ""
echo "=========================================="
echo "PHASE 2: Combining Single Agents into MPO"
echo "=========================================="

# Derive model paths from config parameters
# Model naming convention: ia2c_models/t_{num_env_steps}_{parallel_envs}_{n_steps}_{max_ep}_{gamma}_{config_name}
declare -a SOURCE_MODELS=()
for config in "${SINGLE_CONFIGS[@]}"; do
    BASENAME=$(basename "$config" .json)
    NUM_STEPS=$(python3 -c "import json; print(json.load(open('$config'))['algorithm']['num_env_steps'])")
    PAR_ENVS=$(python3 -c "import json; print(json.load(open('$config'))['env']['parallel_envs'])")
    N_STEPS=$(python3 -c "import json; print(json.load(open('$config'))['algorithm']['n_steps'])")
    MAX_EP=$(python3 -c "import json; print(json.load(open('$config'))['env']['max_ep_length'])")
    GAMMA=$(python3 -c "import json; print(json.load(open('$config'))['algorithm']['model']['gamma'])")
    MODEL_PATH="ia2c_models/t_${NUM_STEPS}_${PAR_ENVS}_${N_STEPS}_${MAX_EP}_${GAMMA}_${BASENAME}"
    SOURCE_MODELS+=("$MODEL_PATH")
    echo "  Agent model: $MODEL_PATH"
done

echo ""
echo "Combining into: $COMBINED_MODEL_DIR"

python utils/combine_single_agents.py \
    --source_models "${SOURCE_MODELS[@]}" \
    --source_configs "${SINGLE_CONFIGS[@]}" \
    --target_config "$COMBINED_CONFIG" \
    --output "$COMBINED_MODEL_DIR" \
    2>&1 | tee "$LOGDIR/combine_${TIMESTAMP}.txt"

echo ""
echo "Phase 2 completed at $(date)"

# ============================================================
# Phase 3: Fine-tune combined multi-agent model
# ============================================================

echo ""
echo "=========================================="
echo "PHASE 3: Fine-tuning Combined MPO Model"
echo "=========================================="

echo "Config: $COMBINED_CONFIG"
echo "Pretrained model: $COMBINED_MODEL_DIR"

python train.py --config "$COMBINED_CONFIG" \
    2>&1 | tee "$LOGDIR/train_combined_${TIMESTAMP}.txt"

echo ""
echo "=========================================="
echo "FULL PIPELINE COMPLETED at $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Single-agent models: ia2c_models/t_*"
echo "  Combined pretrained: $COMBINED_MODEL_DIR"
echo "  Fine-tuned model:    ia2c_models/t_*_combined_w_pretrained_fxa_logp_molweight_tpsa"
echo "  Logs:                $LOGDIR/"
echo "=========================================="
