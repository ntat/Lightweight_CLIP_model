#!/bin/bash
#SBATCH --job-name=multi-node-accelerate
#SBATCH --account=account_billing_group
#SBATCH --nodes=2
#SBATCH --gres=gpu:A40:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

# Activate your environment
module load torchvision
source /dsk/nikos/flashattention/bin/activate

# Specify the number of GPUs per node
GPUS_PER_NODE=4
MASTER_ADDR=$(hostname -i)        # Get the IP address of the master node
MASTER_PORT=42069                 # Specify a port for communication

# Set up the launch command for accelerate
export LAUNCH_CMD="
    accelerate launch \
        --multi_gpu --mixed_precision no \
        --num_machines=${SLURM_NNODES} \
        --num_processes=$(expr ${SLURM_NNODES} \* ${GPUS_PER_NODE}) \
        --machine_rank=\${SLURM_NODEID} \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        main.py \
    "

# Run the launch command on all nodes using srun
echo ${LAUNCH_CMD}
srun bash -c "${LAUNCH_CMD}"