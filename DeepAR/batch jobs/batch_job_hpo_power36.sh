#!/bin/bash
#SBATCH --job-name=da_hop_power_36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_hop_test.txt
#SBATCH --error=job_err_power_hop_test.txt

module load gcc/8.3.0 cuda/10.1.168
module load python-data
module list

#cd DeepAR

cd ..

echo "Installing requirements ..."

pip3 install -r 'requirements.txt' -q --user
pip3 install hyperopt -q --user

echo "requirements installed."

echo  'load the data'

python3 preprocess_power_system.py --dataset='elect' \
                                  --data-folder='data' \
                                  --model-name='power_system_hop_36_3' \
                                  --file-name='europe_power_system.csv' \
                                  --hop

echo "data loaded."
echo "Search hyperparameters ..."

python3 search_hyperparams_custom.py --sampling \
                                     --dataset='elect' \
                                     --model-name='power_system_hop_36_3' \
                                     --evals=50



#load the data

python3 preprocess_elect.py --dataset='elect' \
                            --data-folder='data' \
                            --model-name='power_system_hop_36_3' \
                            --file-name='europe_power_system.csv'

echo 'Training ...'

#{'batch_size': 512.0, 'learning_rate': 0.005, 'lstm_dropout': 0.2, 'lstm_hidden_dim': 256.0}

python3 train.py --sampling \
                 --dataset='elect' \
                 --data-folder='power_system_hop_36_3' \
                 --model-name='power_system_hop_36_3' \
                 --restore-file='epoch_25'

echo 'Testing ... '

#python3 preprocess_elect.py --dataset='elect' \
#                            --data-folder='data' \
#                            --model-name='power_system_hop_36_3' \
#                            --file-name='europe_power_system.csv' \
#                            --test


#python3 evaluate.py --sampling \
#                    --dataset='elect' \
#                    --data-folder='power_system_hop_36_3' \
#                    --model-name='power_system_hop_36_3' \
#                    --save-file \
#                    --restore-file='best'

echo "Finished running!"

seff $SLURM_JOBID
