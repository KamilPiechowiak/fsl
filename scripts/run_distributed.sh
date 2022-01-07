#!/bin/bash
#SBATCH --job-name=fsl
#SBATCH --output=logs/%x-%j_%s.out
#SBATCH --error=logs/%x-%j_%s.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -p lab-ci-student
#SBATCH --exclude=lab-ci-11,lab-ci-2,lab-ci-5,lab-ci-10,lab-ci-14,lab-ci-4,lab-ci-3

srun scripts/run_dist_local.sh $1