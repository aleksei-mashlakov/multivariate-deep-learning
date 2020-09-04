#!/bin/bash
#SBATCH --job-name=tcn_hop
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1,nvme:32
#SBATCH --output=job_out_ep.txt
#SBATCH --error=job_err_ep.txt

module load gcc/8.3.0 cuda/10.1.168
module load python-data
module load mxnet
module list

#cd DeepAR

cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."

echo "Loading data ... "
#srun python3 ec_feature_preprocess_custom.py
echo "Data loaded."

echo "Start running ... "
srun python3 ECPointHuber_HOP.py --dataset="feature_prepare_new2.pkl" \
                                 --save-folder="./save" \
                                 --save-file="hop_train_results.csv" \
                                 --hop-file="hyper_parameter_search.pkl" \
                                 --evals=100 \
                                 --gpu
echo "Finished running!"

seff $SLURM_JOBID
