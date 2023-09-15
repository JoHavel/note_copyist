#!/bin/bash
#SBATCH -J latentSymGen
#SBATCH --array=0-7
#SBATCH --output=out/logs/generate-%A_%a.out
#SBATCH --error=out/logs/generate-%A_%a.err
#SBATCH -p gpu-ms
#SBATCH --gpus=1
#SBATCH --mem=32gb
#SBATCH --exclude=dll-10gpu[2,3]
# exclude to not take up Titans I'll need for the evaluation

# you do:
# sbatch ./slurm-experiments-generate.sh 30

ID_TO_EXPERIMENT="
A
B
C
D
E
F
G
H
"

EXPERIMENT=$(echo "$ID_TO_EXPERIMENT" | head -n $(expr 2 + $SLURM_ARRAY_TASK_ID) | tail -n 1)
SEED=$1

if [ -z "$SEED" ]; then
    echo "Seed argument missing"
    exit 1
fi

echo "################################"
echo "# Generating symbols experiment $EXPERIMENT, seed $SEED"
echo "################################"
echo

export LD_LIBRARY_PATH=/opt/cuda/11.8/lib64:/opt/cuda/11.8/cudnn/8.6.0/lib64

stdbuf -o0 -e0 bash ./experiments_generate.sh $SEED $EXPERIMENT

echo
echo "########"
echo "# DONE #"
echo "########"
