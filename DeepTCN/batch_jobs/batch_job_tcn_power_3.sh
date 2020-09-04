#!/bin/bash
#SBATCH --job-name=tcn_p3
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_33.txt
#SBATCH --error=job_err_power_33.txt

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
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_3_3.pkl' \
                                 --horizon=3
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='power_3_3.pkl' \
                                        --pickle-name='power_system_3_3.pkl' \
                                        --dim=183 \
                                        --horizon=3 \
                                        --patience=50 \
                                        --gpu
echo "Finished running training!"

echo "Loading data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='europe_power_system.csv' \
                                 --pickle-name='power_system_3_3.pkl' \
                                 --horizon=3 \
                                 --test
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='power_3_3.pkl' \
                                                 --pickle-name='power_system_3_3.pkl' \
                                                 --dim=183 \
                                                 --horizon=3 \
                                                 --gpu


seff $SLURM_JOBID
