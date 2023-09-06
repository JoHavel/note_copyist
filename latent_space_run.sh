set -ueo pipefail

SEED=$1
LATENT_SPACE=$2

python main.py -q --seed $SEED --directory out/models_for_mashcima/L${LATENT_SPACE}_${SEED} --dataset other --dataset_dir downloads/MUSCIMA$SEED --cat cat --layers --conv_layers 64 16 4 --stride 3 --kernel 5 --multiply_of 27 --batch 5 --network aae --epoch 150 --latent $LATENT_SPACE
python generate_images_for_mashcima.py ../symbol-synthesis/datasets/latent/L${LATENT_SPACE}_$SEED out/models_for_mashcima/L${LATENT_SPACE}_${SEED}/parts/e150 --network aae --cat cat --seed $SEED
