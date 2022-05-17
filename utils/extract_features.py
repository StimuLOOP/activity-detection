from utils.helper import *
from utils.dataset import *
from utils.moncada_torres_extractor import *

def get_dataset(
    patients,  
    sensors='wrists', 
    label_name='activity',
    w_size=128, 
    w_overlap=64, 
    data_root='/datasets/GaitDetection'
    ):
    # Assemble sensor positions
    valid_sensor_locations = ['wrists', 'ankles', 'all', 'no_chest',
                               'wrist_r', 'wrist_l', 'chest', 'ankle_l', 'ankle_r'
                              ]
    if sensors in ['wrist_r', 'wrist_l', 'chest', 'ankle_l', 'ankle_r']:
        s_positions = [sensors]
    elif sensors == 'wrists':
        s_positions = ['wrist_r', 'wrist_l']
    elif sensors == 'ankles':
        s_positions = ['ankle_r', 'ankle_l']
    elif sensors == 'all':
        s_positions = ['chest', 'wrist_r', 'wrist_l', 'ankle_r', 'ankle_l']
    elif sensors == 'no_chest':
        s_positions = ['wrist_r', 'wrist_l', 'ankle_r', 'ankle_l']
    else:
        raise ValueError(f"Invalid sensors '{sensors}', should be one of {valid_sensor_locations}")
    
    # Get dataset of specified patients
    dataset, labels = [], []
    for patient in patients:
        # Get dataset from feature extractor  
        data, lab = moncada_torres_patient_dataset(
                patient, 
                s_positions, 
                label_name=label_name,
                w_size=w_size, 
                w_overlap=w_overlap, 
                data_root=data_root
            )
        dataset.append(data)
        labels.append(lab)
    return dataset, labels
    