#!/bin/bash
#SBATCH --job-name=tcn_p_load_36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_load_36.txt
#SBATCH --error=job_err_power_load_36.txt

module load gcc/8.3.0 cuda/10.1.168
#module load python-data
module load mxnet


cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."

echo "Loading data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system_load.csv' \
                                 --pickle-name='power_system_load_36.pkl' \
                                 --horizon=36
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='power_load_36.pkl' \
                                        --pickle-name='power_system_load_36.pkl' \
                                        --dim=59 \
                                        --horizon=36 \
                                        --patience=25 \
                                        --gpu


echo "Finished running training!"

echo "Loading data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system_load.csv' \
                                 --pickle-name='power_system_load_36.pkl' \
                                 --horizon=36 \
                                 --test
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='power_load_36.pkl' \
                                                 --pickle-name='power_system_load_36.pkl' \
                                                 --dim=59 \
                                                 --horizon=36 \
                                                 --gpu

echo "Finished running!"

seff $SLURM_JOBID
