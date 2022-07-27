from utils.constants import *
from utils.moncada_torres_extractor import *

def get_data(
    fn,
    sensors='all',
    w_size=128
    ):
    # Assemble sensor positions
    if sensors in ['wrist_l', 'wrist_r']:
        s_positions = [sensors]
    elif sensors == 'left':
        s_positions = ['wrist_l', 'ankle_l']
    elif sensors == 'right':
        s_positions = ['wrist_r', 'ankle_r']
    elif sensors == 'wrists':
        s_positions = ['wrist_r', 'wrist_l']
    elif sensors == 'ankles':
        s_positions = ['ankle_r', 'ankle_l']
    elif sensors == 'all':
        s_positions = ['chest', 'wrist_r', 'wrist_l', 'ankle_r', 'ankle_l']
    elif sensors == 'no_chest':
        s_positions = ['wrist_r', 'wrist_l', 'ankle_r', 'ankle_l']
    else:
        raise ValueError(f"Invalid sensors '{sensors}', should be one of {VALID_SENSOR_POS}")

    # Get data from feature extractor
    data = moncada_torres_patient_dataset(
            fn,
            s_positions,
            w_size=w_size
        )
    return data
