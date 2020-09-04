#!/bin/bash
#SBATCH --job-name=dsap1686
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=15:50:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_ep_power_168_6.txt
#SBATCH --error=job_err_ep_power_168_6.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.2.0
module list

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Start running ... "
srun python3 single_cpu_trainer.py --data_name europe_power_system --n_multiv 183 --window 168 --horizon 6 --batch_size 128 --split_train 0.7004694835680751 --split_validation 0.14929577464788732 --split_test 0.15023474178403756
echo "Finished running!"

seff $SLURM_JOBID
