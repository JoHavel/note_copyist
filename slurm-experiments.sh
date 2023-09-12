#!/bin/bash
#SBATCH -J latentSymGen
#SBATCH --array=72,73,74
#SBATCH --output=out/logs/experiments-%A_%a.out
#SBATCH --error=out/logs/experiments-%A_%a.err
#SBATCH -p gpu-ms
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --exclude=dll-10gpu[2,3]
# exclude to not take up Titans I'll need for the evaluation

# you do:
# sbatch ./slurm-experiments.sh REBELO|MIX

MODEL_KIND=$1
SEED=$SLURM_ARRAY_TASK_ID

if [ -z "$SEED" ]; then
    echo "Seed argument missing"
    exit 1
fi

echo "################################"
echo "# Latent symbols kind $MODEL_KIND, seed $SEED"
echo "################################"
echo

export LD_LIBRARY_PATH=/opt/cuda/11.8/lib64:/opt/cuda/11.8/cudnn/8.6.0/lib64

stdbuf -o0 -e0 bash ./experiments_train.sh $SEED $MODEL_KIND

echo
echo "########"
echo "# DONE #"
echo "########"
