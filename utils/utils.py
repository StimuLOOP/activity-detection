import ntpath
import datetime
import pandas as pd
from utils.constants import *
from joblib import load
import numpy as np

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def log(msg):
    with open('log.txt', 'a') as fp:
        timestamp =  datetime.datetime.now()
        fp.write(f"[{str(timestamp)}] {msg}\n")

def load_csv(fn):
    # Assemble dataset of the form (num_samples, num_measurements, w_size)
    df = pd.read_csv(fn,on_bad_lines='skip')
    for key in df.keys():
        if not 'prediction' in key:
            df[key] = pd.to_numeric(df[key],errors='coerce')
    df = df.dropna()
    return df

def make_predictions(model_fn, data, task, w_size=128):
    model = load(model_fn)
    means = np.repeat(model[0].mean_.reshape(1,-1), data.shape[0], axis=0)
    data[data==np.inf] = means[data==np.inf]
    curr_preds = model.predict(data)
    curr_preds = np.array([PRED_TO_STRING[task][label] for label in curr_preds]).repeat(w_size)
    return curr_preds