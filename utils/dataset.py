import h5py
import numpy as np
import os
import csv
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import pywt
from sklearn.preprocessing import normalize
from scipy.special import entr
from scipy import signal
from skimage.restoration import denoise_wavelet
from utils.helper import *
import sklearn_relief as relief
    

def get_data(patient, measurement, s_position, label_name='activity', filter_func=None, data_root='/datasets/GaitDetection'):
    '''
    Extract certain measurement from patient and specified position
    '''
    assert (type(patient) == int), 'Pass patient ID as integer'
    
    # Standardize position string
    s_position = standardize_position_name(s_position)
    if s_position not in ['wrist_r', 'wrist_l', 'ankle_r', 'ankle_l', 'chest']:
        raise ValueError(f"Invalid position {s_position}, should be one of ['wrist_r', 'wrist_l', 'ankle_r', 'ankle_l', 'chest']")
    
    # Open mat file
    data_fn = os.path.join(data_root, f'data/P{f"0{patient}" if patient <10 else patient}.mat')
    data = h5py.File(data_fn)
    
    # Get sensor position id
    p_id = get_position_id(data, s_position)
    
    
    # Get measurement
    meas = get_measurement_from_sensor(data, measurement, p_id)
    
    # Patient 6 has recordings on two different days
    if patient ==6:
        data_fn = os.path.join(data_root, f'data/P06a.mat')
        data_6 = h5py.File(data_fn)
        p_id_6 = get_position_id(data_6, s_position)
        meas_6 = get_measurement_from_sensor(data_6, measurement,p_id_6)
        sens_start_6 = ''.join([chr(c[0]) for c in data_6['jumpExp']['header']['startStr']])
        sens_start_6 = dt.datetime.strptime(sens_start_6, '%Y/%m/%d %H:%M:%S.%f')
        
    # Open file with label start/end timestamps
    fn = os.path.join(data_root, f'data/Start_stop_final.xlsx')
    df = pd.read_excel(fn, engine='openpyxl')
    df = df[df['ID'].str.contains(f'p{f"0{patient}" if patient <10 else patient}')]
    df = df[pd.notnull(df['sens_start'])]
    df = df[df['ID']!='p11_03'] # ignore p11_03 as it contains faulty start/stop times
    if len(df)==0:
        raise Exception(f"Video start/stop times not yet defined for patient {patient}")

    # Cut video according to start/end timestamps
    sens_start = ''.join([chr(c[0]) for c in data['jumpExp']['header']['startStr']])
    sens_start = dt.datetime.strptime(sens_start, '%Y/%m/%d %H:%M:%S.%f')
    cut_data = []
    for _, row in df.iterrows():
        if patient == 6 and row['sens_start'] >= sens_start_6:
            # Handle patient 6 separately because they have two separate recordings
            label_start = get_frame_from_timestamp(sens_start_6, row['sens_start'])
            label_stop = get_frame_from_timestamp(sens_start_6, row['sens_stop'])
            cut_data.append(meas_6[:,label_start:label_stop])
        else:
            label_start = get_frame_from_timestamp(sens_start, row['sens_start'])
            label_stop = get_frame_from_timestamp(sens_start, row['sens_stop'])
            cut_data.append(meas[:,label_start:label_stop])
    meas = np.concatenate(cut_data, axis=-1)
    
    # Apply filter if given
    if filter_func is not None:
        meas = filter_func(meas)
    
    # Open label file
    if label_name=='activity':
        label_fn = os.path.join(data_root, f'data/P{f"0{patient}" if patient <10 else patient}_labels.npy')
    else:
        label_fn = os.path.join(data_root, f'data/P{f"0{patient}" if patient <10 else patient}_{label_name}_labels.npy')
    labels = np.load(label_fn)
    labels = get_integer_label(labels, label_name)
    
    # Cutoff unlabeled begin of timeseries
    meas = meas[:,:labels.shape[0]]
    labels = labels[:meas.shape[1]]
    assert(labels.shape[0]==meas.shape[1])
    
    return meas, labels

def sliding_window(patient, measurements, s_positions, label_name='activity', filter_func=[], w_size=128, w_overlap=64, data_root='/datasets/GaitDetection'):
    '''
    Slice timeseries into fixed size windows
    '''
    assert(filter_func == [] or len(filter_func)==len(measurements)), f"filter_func must be [] or of the same length as measurements, got {filter_func}"
    # Assemble dataset of the form (num_samples, num_measurements, w_size)
    proc_data = {}
    for pos in s_positions:
        # Compute sliding window over input data
        proc_data[pos] = {}
        for i, measurement in enumerate(measurements):
            # Get data of current measurement
            if filter_func != []:
                data, _ = get_data(patient, measurement, pos, label_name = label_name, filter_func=filter_func[i], data_root=data_root)

            # Split array into fixed size windows
            num_windows = data.shape[1]//w_size
            pruned_data = data[:, :w_size*num_windows]
            split_data = np.array(np.split(pruned_data,num_windows, axis=1))

            # Split array into fixed size windows for overlap
            data = data[:,w_overlap:]
            num_windows = data.shape[1]//w_size
            pruned_data = data[:, :w_size*num_windows]
            split_data = np.concatenate((split_data, np.array(np.split(pruned_data,num_windows, axis=1))),axis=0)
            proc_data[pos][measurement] = split_data
            
    # Assemble label per window
    _, labels = get_data(patient, measurements[0], s_positions[0], label_name = label_name, data_root=data_root)
    
    # Split array into fixed size windows
    num_windows = labels.shape[0]//w_size
    pruned_labels = labels[:w_size*num_windows]
    split_labels = np.array(np.split(pruned_labels, num_windows))
    
    # Split array into fixed size windows for overlap
    labels = labels[w_overlap:]
    num_windows = labels.shape[0]//w_size
    pruned_labels = labels[:w_size*num_windows]
    split_labels = np.concatenate((split_labels,np.array(np.split(pruned_labels, num_windows))))
    
    # Assign majority label to each window
    labels = []
    for i in range(split_labels.shape[0]):
        unique_labels, unique_counts = np.unique(split_labels[i,:], return_counts=True)
        majority_label_idx = np.argmax(unique_counts)
        labels.append(unique_labels[majority_label_idx])
    labels = np.array(labels)
        
    # Relabel stand to walk transition
    labels = relabel_stand_to_walk(labels)
    labels = relabel_walk_to_stairs(labels)
        
    return proc_data, labels

def relabel_stand_to_walk(labels):
    """
    Map st_to_wk to walk if before walk label
    """
    trans_start_idx = -1
    for i in range(labels.shape[0]):
        # If transition is encountered
        if labels[i] == -3:
            # Set start index or do nothing if start index already set
            if trans_start_idx==-1:
                trans_start_idx=i
        # If encountering walking after transition index was set 
        # relabel sequence to walking
        elif labels[i] == 3 and trans_start_idx!=-1:
            labels[trans_start_idx:i] = 3
            trans_start_idx = -1
        # If any other label is encountered, transition shouldnt be relabeled
        else:
            trans_start_idx = -1
    return labels
    
def relabel_walk_to_stairs(labels):
    """
    Map wk_to_stairs to stairs
    """
    labels[labels==-4] = 4
    return labels
    
    
    
    
    
    