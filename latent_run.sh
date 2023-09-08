. lib.sh

# Arguments
SEED=$1
LATENT_SPACE=$2

# Body
train_latent "$SEED" "$LATENT_SPACE"
generate_latent "$SEED" "$LATENT_SPACE"
