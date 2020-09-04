#!/bin/bash
#SBATCH --job-name=dsatest2
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:45:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jo_test_runs_ele_mode.txt
#SBATCH --error=je_test_runs_ele_mode.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list


echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Start running ... "
srun python3 test_electricity.py
echo "Finished running!"

seff $SLURM_JOBID
