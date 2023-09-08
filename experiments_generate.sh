. lib.sh

# Arguments
SEED=$1
EXPERIMENT=$2

# Body

# comment for usage without running latent part
if [ "$EXPERIMENT" == "E" ]; then
  rm -r "$(EXPERIMENT_OUTPUT "$SEED" E)" || :
  cp -r "$(LATENT_OUTPUT "$SEED" "$FINAL_LATENT")" "$(EXPERIMENT_OUTPUT "$SEED" E)"
  exit 0
fi

"generate_${EXPERIMENT}" "$SEED"
