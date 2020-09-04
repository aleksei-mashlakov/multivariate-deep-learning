#!/bin/bash
#SBATCH --job-name=da_hop_elect_36
#SBATCH --account=Project_2002244
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_elect_hop_test.txt
#SBATCH --error=job_err_elect_hop_test.txt

module load gcc/8.3.0 cuda/10.1.168
module load python-data
module list

#cd DeepAR

cd ..

echo "Installing requirements ..."

pip3 install -r 'requirements.txt' -q --user
pip3 install hyperopt -q --user

echo "requirements installed."

seff $SLURM_JOBID


echo  'load the data'

python3 preprocess_elect.py --dataset='elect' \
                           --data-folder='data' \
                           --model-name='base_model_hop_36_3' \
                           --file-name='electricity.csv' \
                           --hop

echo "data loaded."
echo "Search hyperparameters ..."

python3 search_hyperparams_custom.py --sampling \
                                     --dataset='elect' \
                                     --model-name='base_model_hop_36_3' \
                                     --evals=30

echo 'Load train data'

python3 preprocess_elect.py --dataset='elect' \
                           --data-folder='data' \
                           --model-name='base_model_hop_36_3' \
                           --file-name='electricity.csv'

# echo 'Train data loaded'

# echo 'Training ...'
#{'batch_size': 128.0, 'learning_rate': 0.005, 'lstm_dropout': 0.30000000000000004, 'lstm_hidden_dim': 256.0}

# python3 train.py --sampling \
#                  --dataset='elect' \
#                  --data-folder='base_model_hop_36_3' \
#                  --model-name='base_model_hop_36_3' \
#                  --restore-file='epoch_19'



echo 'Load test data'

python3 preprocess_elect.py --dataset='elect' \
                            --data-folder='data' \
                            --model-name='base_model_hop_36_3' \
                            --file-name='electricity.csv' \
                            --test

echo 'Test data loaded'

echo 'Testing ...'

python3 evaluate.py --sampling \
                    --dataset='elect' \
                    --data-folder='base_model_hop_36_3' \
                    --model-name='base_model_hop_36_3' \
                    --save-file \
                    --restore-file='best'


echo "Finished running!"

seff $SLURM_JOBID
