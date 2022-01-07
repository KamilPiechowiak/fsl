#!/bin/bash

export MASTER_PORT=12340
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "NODE_ID=$SLURM_NODEID"
echo "TOTAL_TASKS=$SLURM_JOB_NUM_NODES"
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_LOCALID=$SLURM_LOCALID"
echo "SLURM_NODEID=$SLURM_NODEID"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "SLURMD_NODENAME=$SLURMD_NODENAME"

docker build -t iatransfer-pytorch scripts/docker
docker rm -f fsl || true
nvidia-docker run --rm --name fsl --ipc=host --user 16023 -p 12340:12340 --network host \
    -e MASTER_PORT=$MASTER_PORT \
    -e MASTER_ADDR=$MASTER_ADDR \
    -v /home/inf136780/fsl:/workspace/fsl \
    iatransfer-pytorch bash -c "cd fsl && NCCL_DEBUG=INFO CUDA_LAUNCH_BLOCKING=1 python3 -m src.run $1 $SLURM_PROCID $SLURM_JOB_NUM_NODES"