#!/bin/bash
#SBATCH --job-name=dsae128_3
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=jo_ele_128_3.txt
#SBATCH --error=je_ele_128_3.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.2.0
module list


echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Start running ... "
srun python3 single_gpu_trainer_electricity.py --data_name electricity --n_multiv 321 --window 128 --horizon 3 --batch_size 64 --split_train 0.6003649635036497 --split_validation 0.19981751824817517 --split_test 0.19981751824817517
echo "Finished running!"

seff $SLURM_JOBID
