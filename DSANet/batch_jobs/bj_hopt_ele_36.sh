#!/bin/bash
#SBATCH --job-name=dsaehopt
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jo_ele_36_hopt.txt
#SBATCH --error=je_ele_36_hopt.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"

echo "Start running ... "
srun python3 new_hopt_gpu_trainer_electricity.py --data_name electricity --window 168 --horizon 36 --split_train 0.1104014598540146 --split_validation 0.028284671532846715 --split_test 0.028284671532846715
echo "Finished running!"

seff $SLURM_JOBID
