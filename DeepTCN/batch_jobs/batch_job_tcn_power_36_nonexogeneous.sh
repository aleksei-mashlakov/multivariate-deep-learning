#!/bin/bash
#SBATCH --job-name=tcn_p_nonex_36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_nonex_36_test.txt
#SBATCH --error=job_err_power_nonex_36_test.txt

module load gcc/8.3.0 cuda/10.1.168
#module load python-data
module load mxnet


cd ../electricity/NewTCNQuantile

#echo "Installing requirements ..."
#pip install -r 'requirements.txt'
#--user -q --no-cache-dir
#echo "Requirements installed."

gpuseff $SLURM_JOBID

echo "Loading data ... "
python3 ec_feature_preprocess_nonexogeneous.py --data-folder='data' \
                                                 --file-name='europe_power_system.csv' \
                                                 --pickle-name='power_system_nonex_36.pkl' \
                                                 --horizon=36
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='power_nonex_36.pkl' \
                                        --pickle-name='power_system_nonex_36.pkl' \
                                        --dim=183 \
                                        --horizon=36 \
                                        --patience=25 \
                                        --gpu


echo "Finished running training!"

echo "Loading data ... "
python3 ec_feature_preprocess_nonexogeneous.py --data-folder='data' \
                                                 --file-name='europe_power_system.csv' \
                                                 --pickle-name='power_system_nonex_36.pkl' \
                                                 --horizon=36 \
                                                 --test
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='power_nonex_36.pkl' \
                                                 --pickle-name='power_system_nonex_36.pkl' \
                                                 --dim=183 \
                                                 --horizon=36 \
                                                 --gpu

echo "Finished running!"
