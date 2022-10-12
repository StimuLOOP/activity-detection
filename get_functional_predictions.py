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
    parser.add_argument('-s', '--sensors', default='wrist_r', type=str, choices= VALID_SENSOR_POS_FUNCTIONAL, help='Specify sensor positions to use')

    # Affected flag
    parser.add_argument('-a', '--affected', action='store_true', help='Set this flag if wrist is affected')
    
    # In person normalization flag
    parser.add_argument('-n', '--no_inperson_standardization', action='store_true', help='Set this flag if inperson normalization should be shut off. This is recommended for scenarios where the data collection protocol deviated significantly from the paper. (E.g. longer timeseries, healthy patients, patients with non diverse activities like e.g. only lying)')

    # Output location
    parser.add_argument('-o', '--output_location', type=str, help='Specify output location.')

    # Append prediction
    parser.add_argument('--append', action='store_true', help='Set this flag if output should be appended to input file.')

    return parser

def make_predictions(model_fn, data, task):
    model = load(model_fn)
    curr_preds = model.predict(data)
    curr_preds = np.array([PRED_TO_STRING[task][label] for label in curr_preds]).repeat(100)
    return curr_preds

def main(fn, s_setup, out_loc, affected=False, no_inperson_standardization=False, append=False):
    # Get data from file
    data = get_data(fn, s_setup, w_size=100, patient_standardization=not no_inperson_standardization)

    # Get predictions
    predictions = {}
    task = 'functional'
    if affected:
        setup = 'affected_wrist'
    else:
        setup = 'not_affected_wrist'
    model_fn = os.path.join('models', task, f'{setup}{"_no_in_person_standardized" if no_inperson_standardization else ""}.joblib')
    predictions[f'{task}_{"non" if not affected else ""}affected_predictions'] = make_predictions(model_fn, data, task)

    if not append:
        # Get output location
        out_fn = path_leaf(fn).split('.')[-2]
        out_fn = f'{out_fn}_functional_predictions.csv'
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
            log(f"{fn} contains bad lines, those lines will be skipped and deleted.")
            df = pd.read_csv(fn,on_bad_lines='skip')
        if df.isnull().values.any():
            log(f"{fn} contains invalid values on lines {np.argwhere(df.isnull().values)[:,0]}.")
        df = df.dropna()
        df = pd.concat((df, pd.DataFrame.from_dict(predictions)),axis=1)
        df.to_csv(fn, index=False)
    print(f"Finished computing gait detections for {fn}.")

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    #Get predictions
    main(args.file_path, args.sensors, args.output_location, affected=args.affected, no_inperson_standardization=args.no_inperson_standardization, append=args.append)
