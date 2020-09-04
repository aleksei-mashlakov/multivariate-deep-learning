#!/bin/bash
#SBATCH --job-name=tcn_12
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_elect_12_2.txt
#SBATCH --error=job_err_elect_12_2.txt

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
                                 --file-name='electricity.csv' \
                                 --pickle-name='electricity_12_2.pkl' \
                                 --horizon=12
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting.py --data-folder='data' \
                                        --model-name='elect_12_2.pkl' \
                                        --pickle-name='electricity_12_2.pkl' \
                                        --dim=321 \
                                        --horizon=12 \
                                        --gpu
echo "Finished running!"


echo "Loading data ... "
python3 ec_feature_preprocess.py --data-folder='data' \
                                 --file-name='electricity.csv' \
                                 --pickle-name='electricity_12_2.pkl' \
                                 --horizon=12 \
                                 --test
echo "Data loaded."

echo "Start running ... "
python3 ec_probabilistic_forecasting_evaluate.py --data-folder='data' \
                                                 --model-name='elect_12_2.pkl' \
                                                 --pickle-name='electricity_12_2.pkl' \
                                                 --dim=321 \
                                                 --horizon=12 \
                                                 --gpu
echo "Finished running!"

seff $SLURM_JOBID
