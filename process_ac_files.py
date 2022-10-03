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
    # Read files in folder
    files_in_folder = os.listdir(path_to_files)
    
    ################################################################
    # TODO JOHANNES: 
    # Name of the files to save to
    ################################################################
    FILE_NAME = '<prediction condition>'
    
    ################################################################
    # TODO JOHANNES:
    # Name of the outcome column headers
    ################################################################
    OUTCOME_HEADERS = ['outcome_x', 'outcome_y'] # Add any additional column headers here (probably one per outcome)
    COL_HEADERS = ['Pat_id', 'measurements', 'setup', 'mean_AC_aff', 'mean_AC_nonaff'] # Do not change this line
    COL_HEADERS += OUTCOME_HEADERS
    
    # Dictionary for dataframes
    final_dfs = {}
    
    # Iterate over the three days
    for t_id in tqdm(range(3)):
        t_id = f'T{t_id+1}'
        final_dfs[t_id] = {}
        for col_name in COL_HEADERS:
            final_dfs[t_id][col_name] = []
        
        
        # Iterate through all patients
        for p_id in tqdm(range(1000), leave=False):
            if p_id <10:
                p_id = f'00{p_id}'
            elif p_id <100:
                p_id = f'0{p_id}'
            p_id = str(p_id)

            # Iterate through all measurements
            for m_id in range(5):
                m_id = f'm{m_id+1}'

                # Iterate across setups
                for setup in ["wrists","all", "no_chest"]:
                    for aff_side, nonaff_side in [['left', 'right'],['right', 'left']]:
                        fn = f"{p_id}_{m_id}_{t_id}_{setup}_sensors_{aff_side}_aff_AC.csv"
                        fn = os.path.join(path_to_files, fn)
                        # If missing file, write Patient id, measurement id and setup and leave everything else blank
                        if not os.path.exists(fn):
                            if True not in [p_id in csv_name for csv_name in files_in_folder]:
                                continue
                            final_dfs[t_id]['Pat_id'].append(p_id)
                            final_dfs[t_id]['measurements'].append(m_id)
                            final_dfs[t_id]['setup'].append(setup)
                            for col_name in COL_HEADERS:
                                if col_name not in ['Pat_id','measurements','setup']:
                                    final_dfs[t_id][col_name].append('')
                            continue
                        df = pd.read_csv(fn)
                        
                        # Initialize lists to compute AC means
                        mean_AC_aff = []
                        mean_AC_nonaff = []
                        mean_outcomes = {}
                        for outcome in OUTCOME_HEADERS:
                            mean_outcomes[outcome] = []
                        
                        for i in range(len(df)):
                            # Read relevant information from csv
                            curr_row = df.iloc[i]
                            activity_prediction = curr_row['activity_prediction']
                            gait_prediction = curr_row['gait_prediction']
                            functional_affected_predictions = curr_row['functional_affected_predictions']
                            functional_nonaffected_predictions = curr_row['functional_affected_predictions']
                            activity_counts_affected = curr_row['AC_aff']
                            activity_counts_nonaffected = curr_row['AC_nonaff']
                            ################################################################
                            # TODO JOHANNES:
                            # put prediction conditions here (used example from the mail)
                            ################################################################
                            if gait_predictions == 'no-gait':
                                if functional_affected_predictions == 'functional':
                                    # Add AC to compute mean later
                                    mean_AC_aff.append(activity_counts_affected)
                                    mean_AC_nonaff.append(activity_counts_nonaffected)
                                    
                                    for outcome_name in OUTCOME_HEADERS:
                                        ################################################################
                                        # START OUTCOME COMPUTATION
                                        ################################################################
                                        
                                        ################################################################
                                        # TODO JOHANNES:
                                        # Compute outcomes here.
                                        #
                                        # Note that computed outcome always needs to be stored 
                                        # in "outcome" variable.
                                        # 
                                        # Create one "if" block for every outcome that 
                                        # should be computed.
                                        ################################################################
                                        if outcome_name == 'outcome_x':
                                            outcome = activity_counts_affected+activity_counts_nonaffected
                                        if outcome_name == 'outcome_y':
                                            outcome = activity_counts_affected-activity_counts_nonaffected
                                        ################################################################
                                        # END OUTCOME COMPUTATION
                                        ################################################################
                                        mean_outcome[outcome].append(outcome)
                            
                            # Compute mean for every outcome and activity counts
                            final_dfs[t_id]['mean_AC_aff'].append(np.mean(mean_AC_aff))
                            final_dfs[t_id]['mean_AC_nonaff'].append(np.mean(mean_AC_nonaff))
                            for outcome_name in OUTCOME_HEADERS:
                                final_dfs[t_id][outcome_name].append(np.mean(mean_outcome[outcome]))
    # Save final dataframes to csvs
    return final_dfs
                                
                            

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    #Get predictions
    main(args.file_path)