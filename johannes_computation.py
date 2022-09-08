from get_activity_predictions import main as activity_predictions
from get_gait_predictions import main as gait_predictions
from get_functional_predictions import main as functional_predictions
import os, sys

if __name__ == '__main__':
    file_loc = os.path.abspath(sys.argv[-1])
    for folder in ['wrists','all', 'affected', 'nonaffected', 'no_chest']:
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
            activity_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
            # Compute gait
            gait_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
            # Compute functional
            functional_predictions(file_path, setup, out_loc=None, affected=affected, append=True)
