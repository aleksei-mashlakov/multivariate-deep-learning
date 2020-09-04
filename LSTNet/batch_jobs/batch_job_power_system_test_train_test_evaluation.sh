#!/bin/bash
# created: Nov 24, 2019 5:07 PM
# author: mashlakov

#!/bin/bash
#SBATCH --account=Project_2002244
#SBATCH --job-name=lstnet_power_system_ex
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_ex.txt
#SBATCH --error=job_err_power_ex.txt


module purge
module load python-data/3.7.3-1
module load tensorflow/2.0.0

cd ..

echo "Hola el patron!"
#pip3 install --upgrade pip3 --user
#pip3 install hyperopt --user -q
#pip3 install -r 'requirements.txt' -q --user
#srun python3 preprocess_elect_custom.py

# Train and test the rest of the horizons
python3 main.py --data="data/europe_power_system.txt" \
                 --horizon=3 \
                 --save="save/power_system_save_3/power_system_3"\
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
                 --save-plot="save/power_system_save_3/results_power_system_3"\
                 --logfilename="log/lstnet_power_system_save_3"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system.txt" \
                 --horizon=6 \
                 --save="save/power_system_save_6/power_system_6"\
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
                 --save-plot="save/power_system_save_6/results_power_system_6"\
                 --logfilename="log/lstnet_power_system_save_6"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system.txt" \
                 --horizon=12 \
                 --save="save/power_system_save_12/power_system_12"\
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
                 --save-plot="save/power_system_save_12/results_power_system_12"\
                 --logfilename="log/lstnet_power_system_save_12"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system.txt" \
                 --horizon=24 \
                 --save="save/power_system_save_24/power_system_24"\
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
                 --save-plot="save/power_system_save_24/results_power_system_24"\
                 --logfilename="log/lstnet_power_system_save_24"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system.txt" \
                 --horizon=36 \
                 --save="save/power_system_save_36/power_system_36"\
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
                 --save-plot="save/power_system_save_36/results_power_system_36"\
                 --logfilename="log/lstnet_power_system_save_36"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


# Train and test the rest of the horizons

python3 main.py --data="data/europe_power_system.txt" \
                --horizon=3 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_power_3_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_save_3/power_system_3" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system.txt" \
                --horizon=6 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_power_6_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_save_6/power_system_6" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system.txt" \
                --horizon=12 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_power_12_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_save_12/power_system_12" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system.txt" \
                --horizon=24 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_power_24_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_save_24/power_system_24" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_save_power_36_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_save_36/power_system_36" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \

echo "Adios el patron!"

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
