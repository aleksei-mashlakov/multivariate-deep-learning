#!/bin/bash
#SBATCH --job-name=tcn_p36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_hop_36_4.txt
#SBATCH --error=job_err_power_hop_36_4.txt

module load gcc/8.3.0 cuda/10.1.168
#module load python-data
module load mxnet


cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip3 install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."

gpuseff $SLURM_JOBID

echo "Loading data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_hop_36_3.pkl' \
                                 --horizon=36 \
                                 --hop
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_hpo.py --data-folder='data' \
                                --model-name='power_hop_36_3.pkl' \
                                --pickle-name='power_system_hop_36_3.pkl' \
                                --dim=183 \
                                --horizon=36 \
                                --save-folder="./save" \
                                --save-file="power_system_hop_36_3_results.csv" \
                                --hop-file="power_system_hop_36_hyper_parameter_search_3.pkl" \
                                --evals=100 \
                                --epochs=100 \
                                --patience=5 \
                                --gpu

# echo "Finished running!"


echo "Loading data for training ... "

python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_hop_36_3.pkl' \
                                 --horizon=36
echo "Training data loaded."

echo "Start running training... "

#Optimization parameters {'batch_size': 64.0, 'dropout': 0.4, 'learning_rate': 0.005, 'units': 64.0}

python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='power_hop_36_3.pkl' \
                                        --pickle-name='power_system_hop_36_3.pkl' \
                                        --dim=183 \
                                        --horizon=36 \
                                        --gpu \
                                        --batch_size=64\
                                        --units=64 \
                                        --learning_rate=0.005 \
                                        --dropout=0.4
echo "Finished training!"


echo "Loading testing data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_hop_36_3.pkl' \
                                 --horizon=36 \
                                 --test

echo "Testing data loaded."

echo "Start running testing... "

python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='power_hop_36_3.pkl' \
                                                 --pickle-name='power_system_hop_36_3.pkl' \
                                                 --dim=183 \
                                                 --horizon=36 \
                                                 --gpu
echo "Finished running!"




seff $SLURM_JOBID
