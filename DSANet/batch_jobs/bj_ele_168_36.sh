#!/bin/bash
#SBATCH --job-name=dsae168_36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --output=jo_ele_168_36_calendar.txt
#SBATCH --error=je_ele_168_36_calendar.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.3.0
module list


echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Start running ... "
srun python3 single_gpu_trainer_electricity.py --data_name electricity --n_multiv 327 --window 168 --horizon 36 --batch_size 64 --split_train 0.6003649635036497 --split_validation 0.19981751824817517 --split_test 0.19981751824817517
echo "Finished running!"

seff $SLURM_JOBID
