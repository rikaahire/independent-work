#!/bin/bash
#SBATCH --job-name=train                  # Job name
#SBATCH --output=train_output.log         # Standard output and error log
#SBATCH --error=error.log                 # Error log
#SBATCH --nodes=1                         # Run on a single node
#SBATCH --ntasks=1                        # Total number of tasks
#SBATCH --cpus-per-task=8                 # CPU cores per task
#SBATCH --mem=32G                         # Total memory per node
#SBATCH --time=24:00:00                   # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --mail-type=END,FAIL              # Send email on job completion or failure
#SBATCH --mail-user=sa3937@princeton.edu

# WARNING: Set GPU_NUM to available GPU on the server in CUDA_VISIBLE_DEVICES=<GPU_NUM>
# or remove this flag entirely if only one GPU is present on the device.

# NOTE: If you run into OOM issues, try reducing --num_envs

module load anaconda3/2024.6
conda activate contrastive_rl

export WANDB_MODE=offline

env=ant

for seed in 1 ; do
  XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python training.py \
    --project_name test --group_name first_run --exp_name test --num_evals 50 \
    --seed ${seed} --num_timesteps 10000000 --batch_size 256 --num_envs 512 \
    --discounting 0.99 --action_repeat 1 --env_name arm_push_easy \
    --episode_length 1000 --unroll_length 62  --min_replay_size 1000 --max_replay_size 10000 \
    --contrastive_loss_fn infonce_backward --energy_fn l2 \
    --multiplier_num_sgd_steps 1 --log_wandb
  done

echo "All runs have finished."