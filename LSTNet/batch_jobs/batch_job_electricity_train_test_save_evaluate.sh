#!/bin/bash
# created: Nov 24, 2019 5:07 PM
# author: mashlakov

#!/bin/bash
#SBATCH --account=Project_2002244
#SBATCH --job-name=lstnet_test_save
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_elec_3_6_12.txt
#SBATCH --error=job_err_elec_3_6_12.txt


module purge
module load python-data/3.7.3-1
module load tensorflow/2.0.0


cd ..

echo "Hola el patron!"
#pip3 install --upgrade pip3 --user
#pip3 install hyperopt --user -q
#pip3 install -r 'requirements.txt' -q --user
#srun python3 preprocess_elect_custom.py

python3 main.py --data="data/electricity.txt" \
                 --horizon=3 \
                 --save="save/electricity_3_2/electricity_3"\
                 --epochs=500 \
                 --GRUUnits=100 \
                 --lr=0.001\
                 --batchsize=128\
                 --dropout=0.2\
                 --test \
                 --predict="testingdata" \
                 --savehistory \
                 --plot \
                 --series-to-plot='5' \
                 --save-plot="save/electricity_3_2/results_elect_3"\
                 --logfilename="log/lstnet_elect_3_2"\
                 --debuglevel=20 \
                 --mc-iterations=100 \


python3 main.py --data="data/electricity.txt" \
                --horizon=6 \
                --save="save/electricity_6_2/electricity_6"\
                --epochs=500 \
                --GRUUnits=100 \
                --lr=0.001\
                --batchsize=128\
                --dropout=0.2\
                --test \
                --predict="testingdata" \
                --savehistory \
                --plot \
                --series-to-plot='5' \
                --save-plot="save/electricity_6_2/results_elect_6"\
                --logfilename="log/electricity_6_2"\
                --debuglevel=20 \
                --mc-iterations=100 \

python3 main.py --data="data/electricity.txt" \
                 --horizon=12 \
                 --save="save/electricity_12_2/electricity_12"\
                 --epochs=500 \
                 --GRUUnits=100 \
                 --lr=0.001\
                 --batchsize=128\
                 --dropout=0.2\
                 --test \
                 --predict="testingdata" \
                 --savehistory \
                 --plot \
                 --series-to-plot='5' \
                 --save-plot="save/electricity_12_2/results_elect_12"\
                 --logfilename="log/lstnet_electricity_12_2"\
                 --debuglevel=20 \
                 --mc-iterations=100 \

python3 main.py --data="data/electricity.txt" \
                 --horizon=24 \
                 --save="save/electricity_save_24/electricity_24"\
                 --epochs=500 \
                 --GRUUnits=100 \
                 --lr=0.001\
                 --batchsize=128\
                 --dropout=0.2\
                 --test \
                 --predict="testingdata" \
                 --savehistory \
                 --plot \
                 --series-to-plot='5' \
                 --save-plot="save/electricity_save_24/results_elect_24"\
                 --logfilename="log/lstnet_save_elect_24"\
                 --debuglevel=20 \
                 --mc-iterations=100 \

python3 main.py --data="data/electricity.txt" \
                 --horizon=36 \
                 --save="save/electricity_save_36/electricity_36"\
                 --epochs=500 \
                 --GRUUnits=100 \
                 --lr=0.001\
                 --batchsize=128\
                 --dropout=0.2\
                 --test \
                 --predict="testingdata" \
                 --savehistory \
                 --plot \
                 --series-to-plot='5' \
                 --save-plot="save/electricity_save_36/results_elect_36"\
                 --logfilename="log/lstnet_save_elect_36"\
                 --debuglevel=20 \
                 --mc-iterations=100 \



# Train and test the rest of the horizons

python3 main.py --data="data/electricity.txt" \
                --horizon=3 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_elect_3_2_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_3_2/electricity_3" \
                --mc-iterations=100 \


python3 main.py --data="data/electricity.txt" \
                --horizon=6 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_electricity_6_2_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_6_2/electricity_6" \
                --mc-iterations=100 \


python3 main.py --data="data/electricity.txt" \
                --horizon=12 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_electricity_12_2_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_12_2/electricity_12" \
                --mc-iterations=100 \


python3 main.py --data="data/electricity.txt" \
                --horizon=24 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_elect_24_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_save_24/electricity_24" \
                --mc-iterations=100 \



python3 main.py --data="data/electricity.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_elect_36_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/electricity_save_36/electricity_36" \
                --mc-iterations=100 \



echo "Adios el patron!"

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
