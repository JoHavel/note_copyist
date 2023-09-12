#!/bin/bash
#SBATCH -J latentSymGen
#SBATCH --array=2,4,6,8,10
#SBATCH --output=out/logs/latent-%A_%a.out
#SBATCH --error=out/logs/latent-%A_%a.err
#SBATCH -p gpu-ms
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --exclude=dll-10gpu[2,3]
# exclude to not take up Titans I'll need for the evaluation

SEED=$1 # use 72, 73, 74 | 80, 81, 82
DIMENSION=$SLURM_ARRAY_TASK_ID

if [ -z "$SEED" ]; then
    echo "Seed argument missing"
    exit 1
fi

echo "################################"
echo "# Latent symbols dim $DIMENSION, seed $SEED"
echo "################################"
echo

export LD_LIBRARY_PATH=/opt/cuda/11.8/lib64:/opt/cuda/11.8/cudnn/8.6.0/lib64

stdbuf -o0 -e0 bash ./latent_run.sh $SEED $DIMENSION

echo
echo "########"
echo "# DONE #"
echo "########"
