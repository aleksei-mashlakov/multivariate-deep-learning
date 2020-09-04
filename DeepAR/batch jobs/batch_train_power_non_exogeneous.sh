#!/bin/bash
# created: Nov 24, 2019 5:07 PM
# author: mashlakov

#!/bin/bash
#SBATCH --job-name=deepar_power36_nonex
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power36_nonex.txt
#SBATCH --error=job_err_power36_nonex.txt


# module purge
# module load python-env/3.7.4-ml

module load gcc/8.3.0 cuda/10.1.168
module load python-data

cd ..

echo "Hola el patron!"

pip3 install -r 'requirements.txt' -q --user

# load the data
python3 preprocess_power_system.py --dataset='elect' \
                                   --data-folder='data' \
                                   --model-name='power_system_non_exogeneous_36' \
                                   --file-name='europe_power_system.csv' \
                                   --num_covariates=1
#                                    --test \


echo "Training ..."


# Train and test the rest of the horizons 
python3 train.py --sampling \
                 --dataset='elect' \
                 --data-folder='power_system_non_exogeneous_36' \
                 --model-name='power_system_non_exogeneous_36'

echo "Loading the test data ..."


# load the data
python3 preprocess_power_system.py --dataset='elect' \
                                   --data-folder='data' \
                                   --model-name='power_system_non_exogeneous_36' \
                                   --file-name='europe_power_system.csv' \
                                   --test \
                                   --num_covariates=1
                                   
echo "Testing ..."

## Evaluate the models
python3 evaluate.py --sampling \
                    --dataset='elect' \
                    --data-folder='power_system_non_exogeneous_36' \
                    --model-name='power_system_non_exogeneous_36' \
                    --save-file \
                    --restore-file='best'

echo "Adios el patron!"

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID