#!/bin/bash
#
#SBATCH --job-name=vae_mcmc
#SBATCH --output=logs/vae_mcmc_%a.log
#SBATCH --ntasks=1
#SBATCH --time=00:03:30
#SBATCH --mem=500MB
#SBATCH --cpus-per-task=1
#SBATCH --array=0-299

ml gcc/12.3.0 python/3.11.3
source /fred/oz303/avajpeyi/venvs/starccato_jax_venv/bin/activate
python run_mcmc.py $SLURM_ARRAY_TASK_ID --dataset "injection"
