#!/bin/bash
#SBATCH --job-name=dsaphopt
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jo_power_36_hopt.txt
#SBATCH --error=je_power_36_hopt.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
#cd apex
#pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
#cd ..
echo "Requirements installed."

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"

echo "Start running ... "
srun python3 new_hopt_gpu_trainer_power.py --data_name europe_power_system --window 168 --horizon 36 --split_train 0.11267605633802817 --split_validation 0.02910798122065728 --split_test 0.02910798122065728
echo "Finished running!"

seff $SLURM_JOBID
