#!/bin/false Library, not script! Use source(1) or '.'

set -ueo pipefail

FINAL_LATENT=5

# Network #####################################################################
EPOCH=150
BATCH=5
NETWORK="aae"
CONDITIONED="cat"
NETWORK_ARGS="--layers --conv_layers 64 16 4 --stride 3 --kernel 5 --multiply_of 27"

TRAIN_ARGS="-q --cat $CONDITIONED --network $NETWORK --epoch $EPOCH --batch $BATCH $NETWORK_ARGS"
GENERATE_ARGS=" --cat $CONDITIONED --network $NETWORK"
###############################################################################



# Scripts #####################################################################
PYTHON="python3.11"
PREPARE_DATASET_SCRIPT="${PYTHON} padd_muscima.py"
TRAIN_SCRIPT="${PYTHON} main.py"
TRAIN_SCRIPT_WITH_ARGS="${TRAIN_SCRIPT} ${TRAIN_ARGS}"
GENERATE_SCRIPT="${PYTHON} generate_images_for_mashcima.py"
GENERATE_SCRIPT_WITH_ARGS="${GENERATE_SCRIPT} ${GENERATE_ARGS}"  # FIXME: next argument must starts with --
##############################################################################



# Directories #################################################################
DATASETS_ROOT="downloaded"
MUSCIMA_ORIGINAL_DIR="${DATASETS_ROOT}/muscima-exported-symbols"
function PREPARED_MUSCIMA_DATASET {
  local SEED=$1
  echo "${DATASETS_ROOT}/MUSCIMA${SEED}"
}
function PREPARED_MIXED_DATASET {
  local SEED=$1
  echo "${DATASETS_ROOT}/MIX${SEED}"
}

MODELS_DIR="out/models_for_mashcima"
function model_file {
  local MODEL_DIR=$1
  echo "${MODEL_DIR}/parts/e${EPOCH}"
}
function LATENT_MODEL_DIR {
  local SEED=$1
  local LATENT_SPACE=$2
  echo "${MODELS_DIR}/L${LATENT_SPACE}_${SEED}"
}
function EXPERIMENT_MODEL_DIR {
  local SEED=$1
  local DATASET=$2
  echo "${MODELS_DIR}/${DATASET}${SEED}"
}

OUTPUT_ROOT="../symbol-synthesis/datasets"
function LATENT_OUTPUT {
  local SEED=$1
  local LATENT_SPACE=$2
  echo "${OUTPUT_ROOT}/latent/L${LATENT_SPACE}_${SEED}"
}
function EXPERIMENT_OUTPUT {
  local SEED=$1
  local EXPERIMENT=$2
  echo "${OUTPUT_ROOT}/experiments/${EXPERIMENT}_${SEED}"
}
###############################################################################



# Preparation #################################################################
function fix_typo {
  mv "${MUSCIMA_ORIGINAL_DIR}/eight-note-down" "${MUSCIMA_ORIGINAL_DIR}/eighth-note-down" || :
  mv "${MUSCIMA_ORIGINAL_DIR}/eight-note-up" "${MUSCIMA_ORIGINAL_DIR}/eighth-note-up" || :
}

function prepare_muscima {
  local SEED=$1
  local DATASET_DIR
        DATASET_DIR="$(PREPARED_MUSCIMA_DATASET "$SEED")"
  rm "$DATASET_DIR" || :
  $PREPARE_DATASET_SCRIPT "$DATASET_DIR" MUSCIMA --seed "$SEED"
}

function prepare_rebelo {
  local SEED=$1
  :
}

function prepare_mix {
  local SEED=$1
  local DATASET_DIR
        DATASET_DIR="$(PREPARED_MIXED_DATASET "$SEED")"
  rm "$DATASET_DIR" || :
  $PREPARE_DATASET_SCRIPT "$DATASET_DIR" MUSCIMA REBELO --seed "$SEED"
}
###############################################################################



# Training ####################################################################
function train_other {
  local SEED=$1
  local DATASET_DIR=$2
  local MODEL_DIR=$3
  local LATENT_SPACE=$4
  ${TRAIN_SCRIPT_WITH_ARGS} --seed "$SEED" --directory "$MODEL_DIR" --dataset other --dataset_dir "$DATASET_DIR" --latent "$LATENT_SPACE"
}

function train_rebelo {
  local SEED=$1
  local MODEL_DIR=$2
  local LATENT_SPACE=$3
  ${TRAIN_SCRIPT_WITH_ARGS} --seed "$SEED" --directory "$MODEL_DIR" --dataset crebelo --latent "$LATENT_SPACE"
}

function train_latent {
  local SEED=$1
  local LATENT_SPACE=$2
  train_other "$SEED" "$(PREPARED_MUSCIMA_DATASET "$SEED")" "$(LATENT_MODEL_DIR "$SEED" "$LATENT_SPACE")" "$LATENT_SPACE"
}
###############################################################################



# Generating ##################################################################
function generate {
  # Works only for non-GNN or one GNN generate!!!
  local SEED=$1
  local INPUT=$2
  local OUTPUT_DIR=$3
  # HACK!!! with MUSCIMA or REBELO, script can still take --network and --cat (and ignore them)
  $GENERATE_SCRIPT_WITH_ARGS --seed "$SEED" "$OUTPUT_DIR" $INPUT  # FIXME INPUT splitting
}

function generate_latent {
  local SEED=$1
  local LATENT_SPACE=$2
  local MODEL_DIR
        MODEL_DIR="$(LATENT_MODEL_DIR "$SEED" "$LATENT_SPACE")"
  generate "$SEED" "$(model_file "$MODEL_DIR")" "$(LATENT_OUTPUT "$SEED" "$LATENT_SPACE")"
}
###############################################################################



# Experiments #################################################################
function train_MUSCIMA {
  local SEED=$1
  train_other "$SEED" "$(PREPARED_MUSCIMA_DATASET "$SEED")" "$(EXPERIMENT_MODEL_DIR "$SEED" REBELO)" "$FINAL_LATENT"
}
function train_REBELO {
  local SEED=$1
  train_rebelo "$SEED" "$(EXPERIMENT_MODEL_DIR "$SEED" REBELO)" "$FINAL_LATENT"
}
function train_MIX {
  local SEED=$1
  train_other "$SEED" "$(PREPARED_MIXED_DATASET "$SEED")" "$(EXPERIMENT_MODEL_DIR "$SEED" MIX)" "$FINAL_LATENT"
}

function generate_A {
  local SEED=$1
  generate "$SEED" "MUSCIMA" "$(EXPERIMENT_OUTPUT "$SEED" A)"
}
function generate_B {
  local SEED=$1
  generate "$SEED" "REBELO" "$(EXPERIMENT_OUTPUT "$SEED" B)"
}
function generate_C {
  local SEED=$1
  generate "$SEED" "MUSCIMA REBELO" "$(EXPERIMENT_OUTPUT "$SEED" C)"
}
function generate_D {
  local SEED=$1
  local MODEL_DIR
        MODEL_DIR="$(EXPERIMENT_MODEL_DIR "$SEED" REBELO)"
  # FIXME enquote path
  generate "$SEED" "$(model_file "$MODEL_DIR") MUSCIMA" "$(EXPERIMENT_OUTPUT "$SEED" D)"
}
function generate_E {
  local SEED=$1
  local MODEL_DIR
        MODEL_DIR="$(EXPERIMENT_MODEL_DIR "$SEED" MUSCIMA)"
  # FIXME enquote path
  generate "$SEED" "$(model_file "$MODEL_DIR")" "$(EXPERIMENT_OUTPUT "$SEED" E)"
}
function generate_F {
  local SEED=$1
  local MODEL_DIR
        MODEL_DIR="$(EXPERIMENT_MODEL_DIR "$SEED" REBELO)"
  # FIXME enquote path
  generate "$SEED" "$(model_file "$MODEL_DIR")" "$(EXPERIMENT_OUTPUT "$SEED" F)"
}
function generate_G {
  local SEED=$1
  local MODEL_DIR
        MODEL_DIR="$(EXPERIMENT_MODEL_DIR "$SEED" MIX)"
  # FIXME enquote path
  generate "$SEED" "$(model_file "$MODEL_DIR")" "$(EXPERIMENT_OUTPUT "$SEED" G)"
}
function generate_H {
  local SEED=$1
  local MODEL_DIR
        MODEL_DIR="$(EXPERIMENT_MODEL_DIR "$SEED" MUSCIMA)"
  # FIXME enquote path
  generate "$SEED" "$(model_file "$MODEL_DIR") REBELO" "$(EXPERIMENT_OUTPUT "$SEED" H)"
}
###############################################################################
