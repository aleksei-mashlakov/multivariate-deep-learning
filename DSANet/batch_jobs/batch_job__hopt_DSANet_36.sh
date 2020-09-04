#!/bin/bash
#SBATCH --job-name=dsaehopt
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=job_out_ep_4_electricity_168_36_hopt_f2.txt
#SBATCH --error=job_err_ep_4_electricity_168_36_hopt_f2.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Start running ... "
srun python3 hopt_gpu_trainer_electricity.py --data_name electricity --window 168 --n_multiv 321 --horizon 36 --split_train 0.1104014598540146 --split_validation 0.028284671532846715 --split_test 0.028284671532846715
echo "Finished running!"

seff $SLURM_JOBID
