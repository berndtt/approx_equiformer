#!/bin/bash

#SBATCH --job-name=approx_equiformer
#SBATCH --time 48:00:00
#SBATCH --partition=genoa-hopper-mli.p
#SBATCH --mem=64000
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# run conda initialisation script
source /hits/basement/mli/bin/activate_conda.sh

export SLURM_TMPDIR=${SLURM_TMPDIR:-/scratch/$USER/$SLURM_JOB_ID}
export TMPDIR=${TMPDIR:-$SLURM_TMPDIR}
mkdir -p "$TMPDIR"

# Forward all to the script
"$@"