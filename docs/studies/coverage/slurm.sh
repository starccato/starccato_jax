#!/bin/bash
#
#SBATCH --job-name=vae_mcmc
#SBATCH --output=logs/download_%a.log
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-100

ml gcc/12.3.0 python/3.11.3
source /fred/oz303/avajpeyi/venvs/starccato_jax_venv/bin/activate
srun python run_mcmc.py $SLURM_ARRAY_TASK_ID
