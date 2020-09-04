#!/bin/bash
#SBATCH --job-name=tcn_p36_evaluate
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_hop_36_train_evaluate.txt
#SBATCH --error=job_err_power_hop_36_train_evaluate.txt

module load gcc/8.3.0 cuda/10.1.168
#module load python-data
module load mxnet


cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."


echo "Loading data for training ... "

python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_hop_36.pkl' \
                                 --horizon=36
echo "Training data loaded."

echo "Start running training... "

python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='power_hop_36.pkl' \
                                        --pickle-name='power_system_hop_36.pkl' \
                                        --dim=183 \
                                        --horizon=36 \
                                        --gpu \
                                        --batch_size=128\
                                        --units=200 \
                                        --learning_rate=0.005 \
                                        --dropout=0.3
echo "Finished training!"


echo "Loading testing data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_hop_36.pkl' \
                                 --horizon=36 \
                                 --test

echo "Testing data loaded."

echo "Start running testing... "
#Optimization parameters {'batch_size': 128, 'dropout': 0.30000000000000004, 'learning_rate': 0.005, 'units': 200}

python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='power_hop_36.pkl' \
                                                 --pickle-name='power_system_hop_36.pkl' \
                                                 --dim=183 \
                                                 --horizon=36 \
                                                 --gpu
echo "Finished running!"

seff $SLURM_JOBID
