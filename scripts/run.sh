#!/bin/bash
#SBATCH --job-name=fsl
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -p lab-ci-student
#SBATCH --exclude=lab-ci-11,lab-ci-2

srun docker build -t iatransfer-pytorch scripts/docker
srun docker rm -f fsl || true
srun nvidia-docker run --rm --name fsl --ipc=host --user 16023 -v /home/inf136780/fsl:/workspace/fsl iatransfer-pytorch bash -c "cd fsl && python3 -m src.run $1 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT"
