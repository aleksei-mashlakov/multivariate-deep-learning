#!/bin/bash
#SBATCH --job-name=dsanet
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH --output=job_out_ep.txt
#SBATCH --error=job_err_ep.txt

module load gcc/8.3.0 cuda/10.1.168
module load pytorch/1.2.0
module list

#cd DeepAR

#cd ./DSANet-master/

echo "Installing requirements ..."
pip install -r 'requirements.txt' --user -q --no-cache-dir
echo "Requirements installed."

echo "Loading data ... "
#srun python3 ec_feature_preprocess_custom.py
echo "Data loaded."

echo "Start running ... "
srun python3 single_cpu_trainer.py --data_name electricity --n_multiv 321 --window 128 --horizon 36
echo "Finished running!"

seff $SLURM_JOBID
