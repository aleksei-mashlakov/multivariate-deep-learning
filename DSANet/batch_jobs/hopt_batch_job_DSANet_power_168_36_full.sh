#!/bin/bash
#SBATCH --job-name=dsap16836
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=h_job_out_ep_power_168_36_hopt_full.txt
#SBATCH --error=h_job_err_ep_power_168_36_hopt_full.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

echo "Start running ... "
srun python3 hopt_gpu_trainer_power_full.py --data_name europe_power_system --n_multiv 183 --window 168 --horizon 36 --batch_size 128 --split_train 0.7004694835680751 --split_validation 0.14929577464788732 --split_test 0.15023474178403756
echo "Finished running!"

seff $SLURM_JOBID
