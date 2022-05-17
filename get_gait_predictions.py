from argparse import ArgumentParser
from utils.constants import *
from utils.data import get_data
from joblib import load
import pandas as pd
import numpy as np
import os
from pathlib import Path

def get_parser():
    # Parse input arguments
    parser = ArgumentParser()
    
    # CSV location
    parser.add_argument('file_path', type=str, help='CSV file to make predictions.')
    
    # Sensor positions
    parser.add_argument('-s', '--sensors', default='all', type=str, choices= VALID_SENSOR_POS_ACTIVITY, help='Specify sensor positions to use')
    
    # Output location
    parser.add_argument('-o', '--output_location', type=str, help='Specify output location.')
    
    return parser

def make_predictions(model_fn, data, task):
    model = load(model_fn)
    curr_preds = model.predict(data)
    curr_preds = np.array([PRED_TO_STRING[task][label] for label in curr_preds]).repeat(128)
    return curr_preds

def main(fn, s_setup, out_loc):
    # Get data from file
    data = get_data(fn, s_setup)
    
    # Get predictions
    task = 'gait'
    predictions = {}
    if s_setup not in ['right', 'left']:
        model_fn = os.path.join('models', task, f'{s_setup}.joblib')
        predictions[task] = make_predictions(model_fn, data, task)
    else:
        for setup in ['affected', 'not_affected']:
            model_fn = os.path.join('models', task, f'{setup}.joblib')
            predictions[f'{task}__{setup}'] = make_predictions(model_fn, data, task)
            
    # Get output location
    out_fn = fn.split('/')[-1].split('.')[-2]
    out_fn = f'{out_fn}_gait_predictions.csv'
    if out_loc is None:
        out_loc = Path(fn).parent.absolute()
        
    # Save predictions
    df = pd.DataFrame.from_dict(predictions)
    df.to_csv(os.path.join(out_loc, out_fn),index=False)
    

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    #Get predictions
    main(args.file_path, args.sensors, args.output_location)