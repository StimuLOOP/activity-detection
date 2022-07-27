from scipy.stats import skew, kurtosis, iqr
from scipy import signal
from scipy.fft import fft
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def sliding_window(fn, measurements, s_positions, filter_func=[], w_size = 128):
    '''
    Slice timeseries into fixed size windows
    '''
    # Assemble dataset of the form (num_samples, num_measurements, w_size)
    df = pd.read_csv(fn)
    proc_data = {}
    for pos in s_positions:
        proc_data[pos] = {}
        for i,measurement in enumerate(measurements):
            # Get measurement of position
            cols = []
            if measurement == 'press':
                cols = f'{pos}__{measurement}'
                data = df[cols].to_numpy().reshape(1,-1)
            else:
                cols = [f'{pos}__{measurement}_x',f'{pos}__{measurement}_y',f'{pos}__{measurement}_z']
                data = df[cols].to_numpy().T

            # Split array into fixed size windows
            num_windows = data.shape[1]//w_size
            pruned_data = data[:, :w_size*num_windows]
            split_data = np.array(np.split(pruned_data,num_windows, axis=1))

            proc_data[pos][measurement] = meas = filter_func[i](split_data)
    return proc_data

def moncada_torres_active_filter(sig):
    median_sig = moncada_torres_median_filter(sig)
    sig = median_sig - moncada_torres_pos_filter(sig)
    return sig

def moncada_torres_pos_filter(sig):
    sig = moncada_torres_median_filter(sig)
    sig = moncada_torres_low_pass_filter(sig)
    return sig

def moncada_torres_low_pass_filter(sig):
    filt = signal.ellip(4, 0.01, 100, 0.3, btype='low', output='sos', fs=50)
    sig = signal.sosfiltfilt(filt, sig)
    return sig

def moncada_torres_median_filter(sig):
    sig = ndimage.median_filter(sig, size=3)
    return sig

def moncada_torres_butter_filter(sig):
    '''
    Low pass filter for altitude signal
    '''
    filt = signal.butter(2, 0.07, btype='low', output='sos', fs=50)
    sig = signal.sosfiltfilt(filt, sig)
    sig = sig-sig.mean()
    return sig

def find_peak_mean(data):
    '''
    Find mean of peak data[:,:,p] defined as data[:,:,p-1]<data[:,:,p]>data[:,:,p+1]
    '''
    cut_data = data[:,:,1:-1]
    greater_than_l_neighbor = cut_data>data[:,:,:-2]
    greater_than_r_neighbor = cut_data<data[:,:,2:]
    peaks = np.where(np.logical_and(greater_than_l_neighbor, greater_than_r_neighbor), cut_data, np.zeros_like(cut_data))
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_peak = peaks.sum(axis=-1)/np.count_nonzero(peaks,axis=-1)
        mean_peak = np.nan_to_num(mean_peak, posinf=data.max(), neginf=data.min())
    return mean_peak

def extract_posture_features(data):
    feat = []

    # Mean
    feat.append(data.mean(axis=-1))

    # Standard Deviation
    feat.append(data.std(axis=-1))

    # Variance
    feat.append(data.var(axis=-1))

    # Inter quartile range
    feat.append(iqr(data, axis=-1))

    # Percentiles
    t_percentiles = np.percentile(data, [3,10,20,97], axis=-1)
    feat += [t_percentiles[0], t_percentiles[0], t_percentiles[0], t_percentiles[0]]

    # Peak to peak amplitude
    feat.append(data.max(axis=-1)-data.min(axis=-1))

    # Mean peak to peak amplitude
    feat.append(find_peak_mean(data)+find_peak_mean(-data))

    # Excessive Kurtosis
    feat.append(kurtosis(data, axis=-1)-3)

    # Root mean squared
    feat.append(np.sqrt(np.mean(data**2, axis=-1)))
    return np.concatenate(feat,axis=-1)


def extract_activity_features(data):
    feat = []
    # Standard Deviation
    feat.append(data.std(axis=-1))

    # Variance
    feat.append(data.var(axis=-1))

    # Inter quartile range
    feat.append(iqr(data, axis=-1))

    # Percentiles
    t_percentiles = np.percentile(data, [3,10,20,97], axis=-1)
    feat += [t_percentiles[0], t_percentiles[1], t_percentiles[2], t_percentiles[3]]

    # Peak to peak amplitude
    feat.append(data.max(axis=-1)-data.min(axis=-1))

    # Mean peak to peak amplitude
    feat.append(find_peak_mean(data)+find_peak_mean(-data))

    # Excessive Kurtosis
    feat.append(kurtosis(data, axis=-1)-3)

    # Root mean squared
    feat.append(np.sqrt(np.mean(data**2, axis=-1)))


    for i in range(int(data.shape[1]/3)):

        # Estimate signal magnitude area
        x,y,z = data[:,i*3,:],data[:,i*3+1,:],data[:,i*3+2,:]
        sma = 1./data.shape[-1] * (x+y+z).sum(axis=-1,keepdims=True)
        feat.append(sma)

        # Estimate cross correlations
        norm_x = x-x.mean(axis=-1,keepdims=True)
        norm_y = y-y.mean(axis=-1,keepdims=True)
        norm_z = z-z.mean(axis=-1,keepdims=True)
        denom_x = np.sqrt((norm_x**2).sum(axis=-1,keepdims=True))
        denom_y = np.sqrt((norm_y**2).sum(axis=-1,keepdims=True))
        denom_z = np.sqrt((norm_z**2).sum(axis=-1,keepdims=True))
        xy = (norm_x*norm_y).sum(axis=-1,keepdims=True)/(denom_x*denom_y)
        xz = (norm_x*norm_z).sum(axis=-1,keepdims=True)/(denom_x*denom_z)
        yz = (norm_z*norm_y).sum(axis=-1,keepdims=True)/(denom_z*denom_y)
        feat += [xy, xz, yz]

    # Frequency domain features
    freq_data = fft(data)

    # Max frequency component
    mags = np.abs(freq_data[:,:,1:])
    max_peak_loc = np.argmax(mags, axis=-1)
    max_freq_comp = 50*(max_peak_loc+1)/freq_data.shape[-1]
    feat.append(max_freq_comp)

    # Compute spectral entropy
    power_spectrum = np.abs(freq_data)**2
    power_spectrum_dist = power_spectrum/power_spectrum.sum(axis=-1, keepdims=True)
    spectral_entropy = -(power_spectrum_dist*np.log(power_spectrum_dist)).sum(axis=-1)
    feat.append(spectral_entropy)

    # Compute spectral energy
    spectral_energy = power_spectrum.sum(axis=-1)/freq_data.shape[-1]
    feat.append(spectral_energy)

    # Compute spectral kurtosis
    feat.append(np.abs(kurtosis(freq_data, axis=-1))-3)

    return np.concatenate(feat,axis=-1)

def extract_gyro_features(data):
    feat = []
    # Standard Deviation
    feat.append(data.std(axis=-1))

    # Variance
    feat.append(data.var(axis=-1))

    # Inter quartile range
    feat.append(iqr(data, axis=-1))

    # Percentiles
    t_percentiles = np.percentile(data, [3,10,20,97], axis=-1)
    feat += [t_percentiles[0], t_percentiles[0], t_percentiles[0], t_percentiles[0]]

    # Peak to peak amplitude
    feat.append(data.max(axis=-1)-data.min(axis=-1))

    # Mean peak to peak amplitude
    feat.append(find_peak_mean(data)+find_peak_mean(-data))

    # Excessive Kurtosis
    feat.append(kurtosis(data, axis=-1)-3)

    # Root mean squared
    feat.append(np.sqrt(np.mean(data**2, axis=-1)))

    # Frequency domain features
    freq_data = fft(data)

    # Compute spectral entropy
    power_spectrum = np.abs(freq_data)**2
    power_spectrum_dist = power_spectrum/power_spectrum.sum(axis=-1, keepdims=True)
    spectral_entropy = -(power_spectrum_dist*np.log(power_spectrum_dist)).sum(axis=-1)
    feat.append(spectral_entropy)

    # Compute spectral energy
    spectral_energy = power_spectrum.sum(axis=-1)/freq_data.shape[-1]
    feat.append(spectral_energy)

    return np.concatenate(feat,axis=-1)

def extract_press_features(data):
    feat = []
    # Standard Deviation
    feat.append(data.std(axis=-1))

    # Variance
    feat.append(data.var(axis=-1))

    # Inter quartile range
    feat.append(iqr(data, axis=-1))

    # Percentiles
    t_percentiles = np.percentile(data, [3,10,20,97], axis=-1)
    feat += [t_percentiles[0], t_percentiles[0], t_percentiles[0], t_percentiles[0]]

    # Peak to peak amplitude
    feat.append(data.max(axis=-1)-data.min(axis=-1))

    # Slope
    slope = (data[:,:,-1]-data[:,:,0])/(data.shape[-1]/50)
    feat.append(slope)

    # Root mean squared
    feat.append(np.sqrt(np.mean(data**2, axis=-1)))

    return np.concatenate(feat,axis=-1)


def moncada_torres_patient_dataset(fn, s_positions, w_size=128):
    '''
    Get data, sequence timeseries into windows and extract features from measurements
    '''
    # Get data
    data_posture = sliding_window(
        fn,
        ['acc'],
        s_positions,
        filter_func = [moncada_torres_pos_filter],
        w_size=w_size
    )
    data = sliding_window(
        fn,
        ['acc','gyro', 'press'],
        s_positions,
        filter_func = [moncada_torres_active_filter, moncada_torres_low_pass_filter, moncada_torres_butter_filter],
        w_size=w_size
    )

    # Get numpy array for each sensor type
    acc_posture, acc_activity, gyro, press = [], [], [], []
    for pos in data:
        acc_posture.append(data_posture[pos]['acc'])
        acc_activity.append(data[pos]['acc'])
        gyro.append(data[pos]['gyro'])
        press.append(data[pos]['press'])
    acc_posture = np.concatenate(acc_posture, axis=1)
    acc_activity = np.concatenate(acc_activity, axis=1)
    gyro = np.concatenate(gyro, axis=1)
    press = np.concatenate(press, axis=1)

    # Extract features for each sensor type
    acc_posture = extract_posture_features(acc_posture)
    acc_activity = extract_activity_features(acc_activity)
    gyro = extract_gyro_features(gyro)
    press = extract_press_features(press)

    # Assemble data
    feat = np.concatenate([acc_posture, acc_activity, gyro, press],axis=-1)

    # Normalize each feature of this patient
    feat = StandardScaler().fit_transform(feat)

    return feat
