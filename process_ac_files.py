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


def main(path_to_files, return_dfs=False):
    # Read files in folder
    files_in_folder = os.listdir(path_to_files)
    
    ################################################################
    # TODO JOHANNES: 
    # Name of the files to save to
    ################################################################
    FILE_NAME = 'outcomes_xyz'
    
    ################################################################
    # TODO JOHANNES:
    # Name of the outcome column headers
    ################################################################
    OUTCOME_HEADERS = ['outcome_x', 'outcome_y'] # Add any additional column headers here (probably one per outcome)
    COL_HEADERS = ['Pat_id', 'measurements', 'setup', 'affected_side', 'mean_AC_aff', 'mean_AC_nonaff'] # Do not change this line
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
            if True not in [p_id in csv_name for csv_name in files_in_folder]:
                continue

            # Iterate through all measurements
            for m_id in range(5):
                m_id = f'm{m_id+1}'

                # Iterate across setups
                for setup in ["wrists","all_sensors", "no_chest", "affected", "nonaffected"]:
                    fn = f"{p_id}_{m_id}_{t_id}_{setup}"
                    fn = os.path.join(path_to_files, fn)
                    
                    # Compute affected side
                    if os.path.exists(f"{fn}_right_aff_AC.csv"):
                        fn = f"{fn}_right_aff_AC.csv"
                        aff_side, nonaff_side = 'right', 'left'
                    elif os.path.exists(f"{fn}_left_aff_AC.csv"):
                        fn = f"{fn}_left_aff_AC.csv"
                        aff_side, nonaff_side = 'left', 'right'
                    # If missing file, write Patient id, measurement id and setup and leave everything else blank
                    else:
                        final_dfs[t_id]['Pat_id'].append(p_id)
                        final_dfs[t_id]['measurements'].append(m_id)
                        for col_name in COL_HEADERS:
                            if col_name not in ['Pat_id','measurements']:
                                final_dfs[t_id][col_name].append(pd.NA)
                        continue
                    
                    # Store measurement infos
                    final_dfs[t_id]['Pat_id'].append(p_id)
                    final_dfs[t_id]['measurements'].append(m_id)
                    final_dfs[t_id]['setup'].append(setup)
                    final_dfs[t_id]['affected_side'].append(aff_side)
                    
                    # Read ac and predictions file
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
                        functional_affected_prediction = curr_row['functional_affected_predictions']
                        functional_nonaffected_prediction = curr_row['functional_nonaffected_predictions']
                        activity_counts_affected = curr_row['AC_aff']
                        activity_counts_nonaffected = curr_row['AC_nonaff']
                        ################################################################
                        # TODO JOHANNES:
                        # put prediction conditions here (used example from the mail)
                        ################################################################
                        if gait_prediction == 'no-gait':
                            if functional_affected_prediction == 'functional':
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
                                    mean_outcomes[outcome_name].append(outcome)

                    # Compute mean for every outcome and activity counts
                    if len(mean_AC_aff) != 0:
                        final_dfs[t_id]['mean_AC_aff'].append(np.mean(mean_AC_aff))
                        final_dfs[t_id]['mean_AC_nonaff'].append(np.mean(mean_AC_nonaff))
                        for outcome_name in OUTCOME_HEADERS:
                            final_dfs[t_id][outcome_name].append(np.mean(mean_outcomes[outcome_name]))
                    else:
                        final_dfs[t_id]['mean_AC_aff'].append(pd.NA)
                        final_dfs[t_id]['mean_AC_nonaff'].append(pd.NA)
                        for outcome_name in OUTCOME_HEADERS:
                            final_dfs[t_id][outcome_name].append(pd.NA)
                            
    # Make dataframes out dicts
    for key, value in final_dfs.items():
         final_dfs[key] = pd.DataFrame.from_dict(value)
    
    # Create average dataframe
    avg_cols = ['mean_AC_aff', 'mean_AC_nonaff'] + OUTCOME_HEADERS
    cat_df = pd.concat([df[avg_cols] for df in final_dfs.values()])
    agg_df = cat_df.groupby(cat_df.index).mean()
    final_dfs['average'] = pd.concat((final_dfs['T1'][['Pat_id','measurements', 'setup', 'affected_side']], agg_df), axis=1)
    
    # Open excel writer
    writer = pd.ExcelWriter(os.path.join(path_to_files,f'{FILE_NAME}.xlsx'), engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    for name, df in final_dfs.items():
        df.to_excel(writer, sheet_name=name)
    writer.save()
    
    if return_dfs:
        return final_dfs
                                
                            

if __name__ == '__main__':
    #Parse arguments
    parser = get_parser()
    args = parser.parse_args()

    #Get predictions
    main(args.file_path)