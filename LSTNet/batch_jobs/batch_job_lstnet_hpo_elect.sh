#!/bin/bash
# created: Nov 24, 2019 5:07 PM
# author: mashlakov

#!/bin/bash
#SBATCH --account=Project_2002244
#SBATCH --job-name=lstnet_hop_elect_36
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_elect_hop_36_test.txt
#SBATCH --error=job_err_elect_hop_36_test.txt

# 32 GB

# module purge
# module load python-env/3.7.4-ml
# module list

module purge
module load python-data/3.7.3-1
module load tensorflow/2.0.0

cd ..

echo "Hola el patron!"
gpuseff $SLURM_JOBID

#pip3 install --upgrade pip3 --user
#pip3 install hyperopt --user -q
#pip3 install -r 'requirements.txt' -q --user
#srun python3 preprocess_elect_custom.py

python3 main.py --data="data/electricity.txt" \
                --horizon=36 \
                --save="save/electricity_hop_36_2/electricity_hop_36" \
                --epochs=100 \
                --optimize \
                --evals=100 \
                --test \
                --predict="testingdata" \
                --savehistory \
                --plot \
                --series-to-plot='5' \
                --save-plot="save/results" \
                --logfilename="log/lstnet_electricity_hop_36_2" \
                --debuglevel=20 \
                --trainpercent=0.1104014598540146 \
                --validpercent=0.028284671532846715 \
                --patience=5

# {'GRUUnits': 128.0, 'batchsize': 128.0, 'dropout': 0.1, 'lr': 0.0005}

python3 main.py --data="data/electricity.txt" \
                 --horizon=36 \
                 --save="save/electricity_hop_36_2/electricity_hop_36"\
                 --epochs=500 \
                 --GRUUnits=128 \
                 --lr=0.0005\
                 --batchsize=128\
                 --dropout=0.1\
                 --test \
                 --predict="testingdata" \
                 --savehistory \
                 --plot \
                 --series-to-plot='5' \
                 --save-plot="save/electricity_hop_36_2/electricity_hop_36"\
                 --logfilename="log/lstnet_electricity_hop_36_2"\
                 --debuglevel=20 \
                 --mc-iterations=100 \

# Train and test the rest of the horizons 

python3 main.py --data="data/electricity.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_electricity_hop_36_2_test"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_hop_36_2/electricity_hop_36" \
                --mc-iterations=100 \
                
#python train_old.py
echo "Adios el patron!"

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
