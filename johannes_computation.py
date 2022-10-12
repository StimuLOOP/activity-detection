from get_activity_predictions import main as activity_predictions
from get_gait_predictions import main as gait_predictions
from get_functional_predictions import main as functional_predictions
from utils.utils import *
from utils.data import get_data
import os, sys, datetime

if __name__ == '__main__':
    file_loc = os.path.abspath(sys.argv[-1])
    for folder in ["wrists","all", "affected", "nonaffected", "no_chest"]:
        for fn in os.listdir(os.path.join(file_loc,folder)):
            # Compute file path
            file_path = os.path.join(file_loc,folder,fn)
            setup = folder
            affected = False
            aff_side = fn.split('_')[-2]
            if 'affected' == folder:
                affected = True
                setup = aff_side
            elif 'nonaffected' == folder:
                if aff_side =='left':
                    setup='right'
                else:
                    setup='left'
                    
            # Get data 
            data = get_data(file_path, setup, patient_standardization=True)
            
            # Compute activity
            log(f"Starting predictions for {file_path}")
            activity_predictions(
                file_path, 
                setup, 
                out_loc=None, 
                affected=affected,
                no_inperson_standardization=True,
                data=data,
                append=True
            )
            log(f"Finished activity predictions for {file_path}")
            # Compute gait
            gait_predictions(
                file_path, 
                setup, 
                out_loc=None, 
                affected=affected, 
                no_inperson_standardization=True,
                data=data,
                append=True
            )
            log(f"Finished gait predictions for {file_path}")

            # Compute functional
            try:
                sensor = 'wrist_r' if aff_side == 'right' else 'wrist_l'
                functional_predictions(
                    file_path, 
                    sensor, 
                    out_loc=None, 
                    affected=True,
                    no_inperson_standardization=True,
                    append=True
                )
            except KeyError as e:
                log(f'Cannot compute functional prediction for affected side, {sensor} not in {fn}')
            try:
                sensor = 'wrist_l' if aff_side == 'right' else 'wrist_r'
                functional_predictions(
                    file_path, 
                    sensor, 
                    out_loc=None, 
                    affected=False,
                    no_inperson_standardization=True,
                    append=True
                )
            except KeyError as e:
                log(f'Cannot compute functional prediction for nonaffected side, {sensor} not in {fn}')
            log(f"Finished functional predictions for {file_path}")
