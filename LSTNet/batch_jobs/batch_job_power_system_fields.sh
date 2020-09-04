#!/bin/bash
# created: Nov 24, 2019 5:07 PM
# author: mashlakov

#!/bin/bash
#SBATCH --account=Project_2002244
#SBATCH --job-name=lstnet_power_groups
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=job_out_power_groups.txt
#SBATCH --error=job_err_power_groups.txt


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
python3 main.py --data="data/europe_power_system_load.txt" \
                 --horizon=36 \
                 --save="save/power_system_load/power_system"\
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
                 --save-plot="save/power_system_load/results_power_system"\
                 --logfilename="log/lstnet_power_system_load"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system_wind.txt" \
                 --horizon=36 \
                 --save="save/power_system_wind/power_system"\
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
                 --save-plot="save/power_system_wind/results_power_system"\
                 --logfilename="log/lstnet_power_system_wind"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system_price.txt" \
                 --horizon=36 \
                 --save="save/power_system_price/power_system"\
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
                 --save-plot="save/power_system_price/results_power_system"\
                 --logfilename="log/lstnet_power_system_price"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system_solar.txt" \
                 --horizon=36 \
                 --save="save/power_system_solar/power_system"\
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
                 --save-plot="save/power_system_solar/results_power_system"\
                 --logfilename="log/lstnet_power_system_solar"\
                 --debuglevel=20 \
                 --mc-iterations=100 \
                 --trainpercent=0.7004694835680751 \
                 --validpercent=0.14929577464788732 \


python3 main.py --data="data/europe_power_system_load.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_power_load_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_load/power_system" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system_wind.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_power_wind_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_wind/power_system" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system_price.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_power_price_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_price/power_system" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \


python3 main.py --data="data/europe_power_system_solar.txt" \
                --horizon=36 \
                --test \
                --no-saveresults \
                --logfilename="log/lstnet_power_solar_eval"\
                --debuglevel=20 \
                --no-train \
                --no-validation \
                --load="save/power_system_solar/power_system" \
                --trainpercent=0.7004694835680751 \
                --validpercent=0.14929577464788732 \
                --mc-iterations=100 \

echo "Adios el patron!"

# This script will print some usage statistics to the
# end of the standard out file
# Use that to improve your resource request estimate
# on later jobs.
seff $SLURM_JOBID
