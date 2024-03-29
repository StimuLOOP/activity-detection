from argparse import ArgumentParser
from utils.constants import *
from utils.data import get_data
from joblib import load
import pandas as pd
import numpy as np
import os
from pathlib import Path
from utils.utils import *

def get_parser():
    # Parse input arguments
    parser = ArgumentParser()

    # CSV location
    parser.add_argument('file_path', type=str, help='CSV file to make predictions.')

    # Sensor positions
    parser.add_argument('-s', '--sensors', default='all', type=str, choices= VALID_SENSOR_POS_ACTIVITY, help='Specify sensor positions to use')

    # Affected flag
    parser.add_argument('-a', '--affected', action='store_true', help='Set this flag if side is affected. Can only be used in conjunction with "left"/"right" sensor setups.')
    
    # In person normalization flag
    parser.add_argument('-n', '--no_inperson_standardization', action='store_true', help='Set this flag if inperson normalization should be shut off. This is recommended for scenarios where the data collection protocol deviated significantly from the paper. (E.g. longer timeseries, healthy patients, patients with non diverse activities like e.g. only lying)')

    # Output location
    parser.add_argument('-o', '--output_location', type=str, help='Specify output location.')

    # Append prediction
    parser.add_argument('--append', action='store_true', help='Set this flag if output should be appended to input file.')

    return parser

def main(fn, s_setup, out_loc, affected=False, no_inperson_standardization=False, data=None, append=False):
    # Get data from file
    if data is None:
        data = get_data(fn, s_setup, patient_standardization=not no_inperson_standardization)

    # Get predictions
    task = 'gait'
    predictions = {}
    if s_setup not in ['right', 'left']:
        model_fn = os.path.join('models', task, f'{s_setup}{"_no_in_person_standardized" if no_inperson_standardization else ""}.joblib')
    else:
        # For unilateral setup, use not_affected by default and affected if configured by user
        if affected:
            setup = 'affected'
        else:
            setup = 'not_affected'
        model_fn = os.path.join('models', task, f'{setup}{"_no_in_person_standardized" if no_inperson_standardization else ""}.joblib')
    predictions[f'{task}_prediction'] = make_predictions(model_fn, data, task)

    if not append:
        # Get output location
        out_fn = path_leaf(fn).split('.')[-2]
        out_fn = f'{out_fn}_gait_predictions.csv'
        if out_loc is None:
            out_loc = Path(fn).parent.absolute()

        # Save predictions
        df = pd.DataFrame.from_dict(predictions)
        Path(out_loc).mkdir(exist_ok=True, parents=True)
        df.to_csv(os.path.join(out_loc, out_fn),index_label='frame_number')
    else:
        # Save predictions
        df = load_csv(fn)
        df = pd.concat((df, pd.DataFrame.from_dict(predictions)),axis=1)
        df.to_csv(fn, index=False)
    print(f"Finished computing functional predictions for {fn}.")

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    if args.affected and args.sensors not in ['right', 'left']:
        raise ValueError('Affected side can only be set for sensor locations "right" and "left"!')

    #Get predictions
    main(args.file_path, args.sensors, args.output_location, affected=args.affected, no_inperson_standardization=args.no_inperson_standardization, append=args.append)
