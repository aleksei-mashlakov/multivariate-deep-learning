import os
import pandas as pd
import csv
import pytorch_lightning as pl
import torch
from model import DSANet
from datetime import datetime


out_file = '/scratch/project_2002244/DSANet/save/test_runs_electricity_final_v2.csv'
ckpt_load_path = '/scratch/project_2002244/DSANet/tb_logs_v2'
path_list = [os.path.join(dirpath, filename) for dirpath, _, filenames in os.walk(ckpt_load_path) for filename
            in filenames if filename.endswith('.ckpt')]

for filename in path_list:
    model = DSANet.load_from_checkpoint(filename)
    trainer = pl.Trainer(resume_from_checkpoint=filename)

    if model.hparams.n_multiv == 321 or model.hparams.n_multiv == 327:
        print('we have electricity data')
    else:
        continue

    if hasattr(model.hparams, 'mcdropout'):
        print("we have mcdropout")
    else:
        print("we set mcdropout to False")
        setattr(model.hparams, 'mcdropout', 'False')

    if hasattr(model.hparams, 'powerset'):
        print("we have powerset")
    else:
        print("we set powerset to all")
        setattr(model.hparams, 'powerset', 'all')

    if hasattr(model.hparams, 'calendar'):
        print("we have calendar")
    else:
        if model.hparams.n_multiv == 189 or model.hparams.n_multiv == 327:
           print("we set calendar to True")
           setattr(model.hparams, 'calendar', 'True')
        else:
           print("we set calendar to False")
           setattr(model.hparams, 'calendar', 'False')
    print(f'data: {model.hparams.data_name}, horizon: {model.hparams.horizon}, window: {model.hparams.window}, powerset: {model.hparams.powerset}, calendar: {model.hparams.calendar}, {filename}')
    try: 
        st_time = datetime.now()
        print(f'Start the test...{st_time}')
        trainer.test(model)
        result = model.test_results
        eval_time = str(datetime.now() - st_time)
        print(f"Test time: {eval_time}")
        print(result)
        of_connection = open(out_file, 'a')
        writer = csv.writer(of_connection)
        writer.writerow([model.hparams.data_name, model.hparams.horizon, model.hparams.window, model.hparams.powerset, model.hparams.calendar, result])
        torch.cuda.empty_cache()
        of_connection.close()
    except Exception as e:
            print(f"we got an exception...: {e}")
            pass

    
