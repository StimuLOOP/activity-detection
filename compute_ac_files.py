import pandas as pd
import os
from utils.activity_counts import *
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

def get_parser():
    # Parse input arguments
    parser = ArgumentParser()

    # CSV location
    parser.add_argument('file_path', type=str, help='Path to files.')

    return parser


def main(path_to_files):

    # Iterate through all patients
    for p_id in tqdm(range(1000)):
        if p_id <10:
            p_id = f'00{p_id}'
        elif p_id <100:
            p_id = f'0{p_id}'

        # Iterate through all measurements
        for m_id in range(5):
            m_id = f'm{m_id+1}'

            # Iterate over the three days
            for t_id in range(3):
                t_id = f'T{t_id+1}'

                # Iterate across setups
                for setup in ["wrists","all", "no_chest"]:
                    for aff_side, nonaff_side in [['left', 'right'],['right', 'left']]:
                        fn = f"{p_id}_{m_id}_{t_id}_{setup}_sensors_{aff_side}_aff"
                        fn = os.path.join(path_to_files, fn)
                        if not os.path.exists(f"{fn}.csv"):
                            continue
                        df = pd.read_csv(f"{fn}.csv")
                        # Skip if shorter than 22 hours
                        if len(df) <= 22*60*60*50:
                            continue
                        res_df = {}
                        for column in ['activity_prediction', 'gait_prediction', 'functional_affected_predictions','functional_nonaffected_predictions']:
                            res_df[column] = agg_labels(df[column])
                        acc = df[[f'wrist_{aff_side[0]}__acc_x',f'wrist_{aff_side[0]}__acc_y',f'wrist_{aff_side[0]}__acc_z']]
                        res_df['AC_aff'] = process_acc(acc.to_numpy().T)
                        acc = df[[f'wrist_{nonaff_side[0]}__acc_x',f'wrist_{nonaff_side[0]}__acc_y',f'wrist_{nonaff_side[0]}__acc_z']]
                        res_df['AC_nonaff'] = process_acc(acc.to_numpy().T)
                        res_df = pd.DataFrame.from_dict(res_df)
                        res_df.to_csv(f'{fn}_AC.csv')

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    #Get predictions
    main(args.file_path)