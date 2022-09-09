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

    # Output location
    parser.add_argument('-o', '--output_location', type=str, help='Specify output location.')

    # Append prediction
    parser.add_argument('--append', action='store_true', help='Set this flag if output should be appended to input file.')

    return parser

def make_predictions(model_fn, data, task):
    model = load(model_fn)
    curr_preds = model.predict(data)
    curr_preds = np.array([PRED_TO_STRING[task][label] for label in curr_preds]).repeat(128)
    return curr_preds

def main(fn, s_setup, out_loc, affected=False, append=False):
    # Get data from file
    data = get_data(fn, s_setup)

    # Get predictions
    task = 'gait'
    predictions = {}
    if s_setup not in ['right', 'left']:
        model_fn = os.path.join('models', task, f'{s_setup}.joblib')
        predictions[f'{task}_prediction'] = make_predictions(model_fn, data, task)
    else:
        # For unilateral setup, use not_affected by default and affected if configured by user
        if affected:
            setup = 'affected'
        else:
            setup = 'not_affected'
        model_fn = os.path.join('models', task, f'{setup}.joblib')
        predictions[f'{task}_prediction__{setup}'] = make_predictions(model_fn, data, task)

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
        try:
            df = pd.read_csv(fn)
        except pd.errors.ParserError as e:
            with open('log.txt', 'a') as log_fp:
                log(log_fp,f"{fn} contains bad lines, those lines will be skipped and deleted.")
            df = pd.read_csv(fn,on_bad_lines='skip')
        if df.isnull().values.any():
            with open('log.txt', 'a') as log_fp:
                log(log_fp, f"{fn} contains invalid values on lines {np.argwhere(df.isnull().values)[:,0]}.")
        df = df.dropna()
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
    main(args.file_path, args.sensors, args.output_location, affected=args.affected, append=args.append)
