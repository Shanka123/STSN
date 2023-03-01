#!/bin/bash


#SBATCH --nodes=2               # node count
#SBATCH --ntasks-per-node=3      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:3             # number of gpus per node
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=slot_attention_autoencoder_transformer_scoring_multigpu_1000weightmse_tcn_16slots_dspritesdecoder_lowerlr_warmup_nolrdecay_morelayers_rowcolposemb_iterations=3_neutral_pgm_run1.log
source ~/.bashrc
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

export NCCL_DEBUG=INFO


srun python train_slot_transformer_pgm_multigpu.py --img_size=80 --batch_size=16 --depth=24 --learning_rate=0.00008 --run='1' --num_epochs=160


