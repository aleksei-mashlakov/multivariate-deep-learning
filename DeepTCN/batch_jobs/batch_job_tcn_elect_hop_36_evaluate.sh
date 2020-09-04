#!/bin/bash
#SBATCH --job-name=tcn_e36_evaluate
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_elect_hop_36_train_evaluate.txt
#SBATCH --error=job_err_elect_hop_36_train_evaluate.txt

module load gcc/8.3.0 cuda/10.1.168
#module load python-data
module load mxnet


cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."

gpuseff $SLURM_JOBID


echo "Loading data for training ... "

python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='electricity.csv' \
                                 --pickle-name='electricity_hop_36.pkl' \
                                 --horizon=36
echo "Training data loaded."

echo "Start running training... "
# Optimization parameters {'batch_size': 1024, 'dropout': 0.2, 'learning_rate': 0.01, 'units': 128}

python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='elect_hop_36.pkl' \
                                        --pickle-name='electricity_hop_36.pkl' \
                                        --dim=321 \
                                        --horizon=36 \
                                        --gpu \
                                        --batch_size=1024 \
                                        --units=128 \
                                        --learning_rate=0.01 \
                                        --dropout=0.2
echo "Finished training!"


echo "Loading testing data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='electricity.csv' \
                                 --pickle-name='electricity_hop_36.pkl' \
                                 --horizon=36 \
                                 --test

echo "Testing data loaded."

echo "Start running testing... "

python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='elect_hop_36.pkl' \
                                                 --pickle-name='electricity_hop_36.pkl' \
                                                 --dim=321 \
                                                 --horizon=36 \
                                                 --gpu
echo "Finished running!"

seff $SLURM_JOBID
