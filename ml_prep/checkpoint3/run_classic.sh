#!/bin/bash
#SBATCH --job-name=antarctic_ml
#SBATCH --account=TG-SEE260003
#SBATCH --partition=shared
#SBATCH --account=uci157
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=classic_pipeline_%j.out
#SBATCH --error=classic_pipeline_%j.err

module load singularitypro

export LOCAL_SCRATCH=/expanse/lustre/projects/uci157/rrogers/temp
export SPARK_LOCAL_DIRS=/expanse/lustre/projects/uci157/rrogers/temp

cd /expanse/lustre/projects/uci157/rrogers

singularity exec \
    --bind /expanse/lustre/projects/uci157/rrogers \
    ~/esolares/singularity_images/spark_py_latest_jupyter_dsc232r.sif \
    python ml_pipeline_classic.py
