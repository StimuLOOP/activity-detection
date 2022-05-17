VALID_SENSOR_POS = [
        'wrists', 
        'ankles', 
        'all', 
        'no_chest',
        'left',
        'right',
        'wrist_l',
        'wrist_r',
    ]

VALID_SENSOR_POS_ACTIVITY = [
        'wrists', 
        'ankles', 
        'all', 
        'no_chest',
        'left',
        'right',
    ]

VALID_SENSOR_POS_FUNCTIONAL = [
        'wrist_l',
        'wrist_r',
    ]

PRED_TO_STRING = {
    'activity':{
        0:'lying',
        1:'sit',
        2:'stand',
        3:'walk',
        4:'stairs',
    },
    'gait':{
        0:'gait',
        1:'no-gait',
    },
    'functional':{
        0: 'non-functional',
        1: 'functional',
    }
}
