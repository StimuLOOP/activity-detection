from get_activity_predictions import main as activity_predictions
from get_gait_predictions import main as gait_predictions
from get_functional_predictions import main as functional_predictions
from utils.utils import *
import os, sys, datetime

if __name__ == '__main__':
    log_fp = open('log.txt', 'a')
    file_loc = os.path.abspath(sys.argv[-1])
    for folder in ["wrists","all", "affected", "nonaffected", "no_chest"]:
        for fn in os.listdir(os.path.join(file_loc,folder)):
            # Compute file path
            file_path = os.path.join(file_loc,folder,fn)
            setup = folder
            affected = False
            if 'affected' == folder:
                affected = True
                setup = fn.split('_')[-2]
            elif 'nonaffected' == folder:
                setup = fn.split('_')[-2]
            # Compute activity
            log(log_fp, f"Starting predictions for {file_path}")
            activity_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
            log(log_fp, f"Finished activity predictions for {file_path}")
            # Compute gait
            gait_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
            log(log_fp, f"Finished gait predictions for {file_path}")

            # Compute functional
            functional_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
            log(log_fp, f"Finished functional predictions for {file_path}")
    log_fp.close()
