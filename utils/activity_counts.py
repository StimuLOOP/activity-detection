'''
Script is heavily based on python implementation on 
https://github.com/jbrond/ActigraphCounts
'''
import math, os
import numpy as np
from scipy import signal
import pandas as pd
import resampy

##predefined filter coefficients, as found by Jan Brond
A_coeff = np.array(
    [1, -4.1637, 7.5712,-7.9805, 5.385, -2.4636, 0.89238, 0.06361, -1.3481, 2.4734, -2.9257, 2.9298, -2.7816, 2.4777,
     -1.6847, 0.46483, 0.46565, -0.67312, 0.4162, -0.13832, 0.019852])
B_coeff = np.array(
    [0.049109, -0.12284, 0.14356, -0.11269, 0.053804, -0.02023, 0.0063778, 0.018513, -0.038154, 0.048727, -0.052577,
     0.047847, -0.046015, 0.036283, -0.012977, -0.0046262, 0.012835, -0.0093762, 0.0034485, -0.00080972, -0.00019623])

def pptrunc(data, max_value):
    '''
    Saturate a vector such that no element's absolute value exceeds max_abs_value.
    Current name: absolute_saturate().
      :param data: a vector of any dimension containing numerical data
      :param max_value: a float value of the absolute value to not exceed
      :return: the saturated vector
    '''
    outd = np.where(data > max_value, max_value, data)
    return np.where(outd < -max_value, -max_value, outd)

def trunc(data, min_value):
  
    '''
    Truncate a vector such that any value lower than min_value is set to 0.
    Current name zero_truncate().
    :param data: a vector of any dimension containing numerical data
    :param min_value: a float value the elements of data should not fall below
    :return: the truncated vector
    '''

    return np.where(data < min_value, 0, data)

def runsum(data, length, threshold):
    '''
    Compute the running sum of values in a vector exceeding some threshold within a range of indices.
    Divides the data into len(data)/length chunks and sums the values in excess of the threshold for each chunk.
    Current name run_sum().
    :param data: a 1D numerical vector to calculate the sum of
    :param len: the length of each chunk to compute a sum along, as a positive integer
    :param threshold: a numerical value used to find values exceeding some threshold
    :return: a vector of length len(data)/length containing the excess value sum for each chunk of data
    '''
    
    N = len(data)
    cnt = int(math.ceil(N/length))

    rs = np.zeros(cnt)

    for n in range(cnt):
        for p in range(length*n, length*(n+1)):
            if p<N and data[p]>=threshold:
                rs[n] = rs[n] + data[p] - threshold

    return rs

def counts(data, filesf, B=B_coeff, A=A_coeff):
    '''
    Get activity counts for a set of accelerometer observations.
    First resamples the data frequency to 30Hz, then applies a Butterworth filter to the signal, then filters by the
    coefficient matrices, saturates and truncates the result, and applies a running sum to get the final counts.
    Current name get_actigraph_counts()
    :param data: the vertical axis of accelerometer readings, as a vector
    :param filesf: the number of observations per second in the file
    :param a: coefficient matrix for filtering the signal, as found by Jan Brond
    :param b: coefficient matrix for filtering the signal, as found by Jan Brond
    :return: a vector containing the final counts
    '''
    
    deadband = 0.068
    sf = 30
    peakThreshold = 2.13
    adcResolution = 0.0164
    integN = 10
    gain = 0.965

    if filesf>sf:
        data = resampy.resample(np.asarray(data), filesf, sf)

    B2, A2 = signal.butter(4, np.array([0.01, 7])/(sf/2), btype='bandpass')
    dataf = signal.filtfilt(B2, A2, data)

    B = B * gain

    #NB: no need for a loop here as we only have one axis in array
    fx8up = signal.lfilter(B, A, dataf)

    fx8 = pptrunc(fx8up[::3], peakThreshold) #downsampling is replaced by slicing with step parameter

    return runsum(np.floor(trunc(np.abs(fx8), deadband)/adcResolution), integN, 0)

def agg_labels(labels):
    # Get window per 1 second period
    num_labels = labels.shape[0]//50
    pruned_labels = labels[:50*num_labels]
    split_labels = np.array(np.split(pruned_labels,num_labels, axis=0))

    # Assign majority label to each window
    tmp_labels = []
    for i in range(split_labels.shape[0]):
        unique_labels, unique_counts = np.unique(split_labels[i,:], return_counts=True)
        majority_label_idx = np.argmax(unique_counts)
        tmp_labels.append(unique_labels[majority_label_idx])
    leftover = labels[50*num_labels:]
    if len(leftover)>0:
        unique_labels, unique_counts = np.unique(leftover, return_counts=True)
        majority_label_idx = np.argmax(unique_counts)
        tmp_labels.append(unique_labels[majority_label_idx])
    return np.array(tmp_labels)

def process_acc(acc):
    # calculate counts per axis
    c1_1s = counts(acc[0], 50)
    c2_1s = counts(acc[1], 50)
    c3_1s = counts(acc[2], 50)
    
    # combine counts in pandas dataFrame
    ac = np.sqrt(c1_1s**2 + c2_1s**2 + c3_1s**2).astype(np.int)
    return ac
#     c_1s = pd.DataFrame(data = {'axis1' : c1_1s, 'axis2' : c2_1s, 'axis3' : c3_1s})
#     c_1s = c_1s.astype(int)
    
#     return c_1s

def main():
    
    '''
    Creates activity counts per second from raw acceleromter data (g-units) 
    This function:
      - reads in data into a pandas dataFrame
      - Calculates activity counts per axis
      - combines the axis in a pandas dataFrame
    :param file: file name of both input and output file
    :param folderInn: directory with input files, containing raw accelerometer data
    :param folderOut: directory with out files, containing activity counts.
    :param filesf: sampling frequency of raw accelerometer data
    :return: none (writes .csv file instead)
    '''
    
    # Get affected sides
    fn = os.path.join('/datasets/GaitDetection/data/affected_side.xlsx')
    affected_df = pd.read_excel(fn, engine='openpyxl')
        
    # Store output csvs in patient dictionary
    top = np.array(['Seconds', 'Arm Affected', 'Affected X Activity Count', 'Affected Y Activity Count', 'Affected Z Activity Count', 'Arm NonAffected', 'NonAffected X Activity Count', 'NonAffected Y Activity Count', 'NonAffectedNonAffected Z Activity Count', 'Pose']).reshape(1,-1)
    out = {}
    num_seconds = {}
    for lab, mode in enumerate(['arm_affected', 'arm_nonaffected']):
        for p_id in tqdm(range(1,15)):
            rel_side = affected_df['side'][p_id-1]
            if mode == 'arm_affected':
                rel_side = 'r' if rel_side == 'right' else 'l'
            else:
                rel_side = 'l' if rel_side == 'right' else 'r'
            # Get activity counts
            acc, curr_labels = get_data(p_id, 'acc', f'wrist_{rel_side}', label_name=mode)
            curr_labels = agg_labels(curr_labels)
            activity_counts = process_acc(acc)
            
            # Determine length of sequence
            if p_id not in num_seconds:
                num_seconds[p_id] = min(len(activity_counts), curr_labels.shape[0])
            activity_counts = np.array(activity_counts.iloc[:num_seconds[p_id]])
            curr_labels = curr_labels[:num_seconds[p_id]].reshape(-1,1)

            # Add labels and activity counts to output
            if lab == 0:
                out[p_id] = np.arange(num_seconds[p_id]).reshape(-1,1)
            out[p_id] = np.concatenate([out[p_id], curr_labels, activity_counts], axis=1)
    
    for p_id in range(1,15):
        _, curr_labels = get_data(p_id, 'acc', f'wrist_r')
        if p_id not in out:
            continue
        curr_labels = agg_labels(curr_labels).reshape(-1,1)
        out[p_id] = np.concatenate([out[p_id], curr_labels], axis=1)
        
    Path('activity_counts').mkdir(exist_ok=True)
    for p_id in out:
        out_rows = np.concatenate([top, out[p_id]], axis = 0)
        np.savetxt(f'activity_counts/p_{p_id}.csv', out_rows, delimiter=',', fmt='%s')
    
if __name__ == '__main__':
    main()