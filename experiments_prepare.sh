. lib.sh

# Arguments
SEED=$1

# Body
fix_typo
prepare_muscima "$SEED" # uncomment for usage without running latent part
prepare_rebelo "$SEED"
prepare_mix "$SEED"

# comment for usage without running latent part
# cp -r "$(LATENT_MODEL_DIR "$SEED" "$FINAL_LATENT")" "$(EXPERIMENT_MODEL_DIR "$SEED" MUSCIMA)"
