import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis, entropy
from scipy.integrate import trapezoid
import antropy as ant
from hurst import compute_Hc
from multitaper_spectogram_python import multitaper_spectrogram


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Define a Butterworth low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low')

# Apply the low-pass filter to data
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Define a Butterworth high-pass filter
def butter_highpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='high')

# Apply the high-pass filter to data
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpower (psd, freqs, fmin, fmax):
    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
    if psd.ndim > 1:
        values = list()
        for i in range(psd.shape[-1]):
            bp = trapezoid(psd[idx_band, i], freqs[idx_band])
            values.append(bp)
        value = np.mean(values)
    else:
        value = trapezoid(psd[idx_band], freqs[idx_band]) 
        
    return round(value, 2)


def compute_bandpower(psd, freqs):
    # Define the frequency bands
    # Delta (0-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-22 Hz), Gamma (22-60 Hz)
    
    return (
        bandpower(psd, freqs, 0, 4),
        bandpower(psd, freqs, 4, 8),
        bandpower(psd, freqs, 8, 13),
        bandpower(psd, freqs, 13, 22),
        bandpower(psd, freqs, 22, 60)
    )



def max_min_distance(segment, window_length=100):
    """
    Calculates the Maximum-Minimum Distance (MMD) over sliding windows.
    
    Parameters:
    - segment: array-like, the signal segment.
    - window_length: int, the length of each sliding window (default is 100).
    
    Returns:
    - mmd: float, the total sum of Maximum-Minimum Distance.
    """
    mmd = 0
    for i in range(0, len(segment) - window_length + 1, window_length):
        window = segment[i:i + window_length]
        max_val = np.max(window)
        min_val = np.min(window)
        distance = np.sqrt((max_val - min_val) ** 2)
        mmd += distance
    return mmd


def energy_sis(segment, fs=250):
    """
    Calculates the EnergySis (Esis) feature of a signal.
    
    Parameters:
    - segment: array-like, the signal segment.
    - fs: int, sampling frequency of the signal (default is 100 Hz).
    
    Returns:
    - esis: float, the EnergySis value for the segment.
    """
    freqs = (
        (0, 4),
        (4, 8),
        (8, 13),
        (13, 22),
        (22, 70),
    )
    wavelength = 100 if len(segment) < 10000 else 1000
    midpoint = np.mean(freqs, axis=1)
    velocity = midpoint * wavelength
    esis = np.sum(np.square(segment)) * velocity
    return esis




def time_domain_features_window(X):
    max_val = np.max(X, axis=-1)
    min_val = np.min(X, axis=-1)
    amplitude = max_val - min_val
    mean = np.mean(X, axis=-1)
    variance = np.var(X, axis=-1)
    skewness = skew(X, axis=-1)
    kurt = kurtosis(X, axis=-1)
    return max_val, min_val, amplitude, mean, variance, skewness, kurt


def complexity_features_window(X):
    #shanon entropy
    hist, _ = np.histogram(X, bins=10)
    shannon_entropy = entropy(hist)
    sampen = ant.sample_entropy(X)
    apen = ant.app_entropy(X)
    H, _, _ = compute_Hc(X, kind='random_walk')
    return shannon_entropy, sampen, apen, H

def complexity_features_all_data(X):
    shannon_entropies = []
    sampens = []
    apens = []
    Hs = []
    for row in X:
        try:
            shannon, sampen, apen, H = complexity_features_window(row)
        except Exception as e:
            shannon = 0
            sampen = 0
            apen = 0
            H = 0
        shannon_entropies.append(shannon)
        sampens.append(sampen)
        apens.append(apen)
        Hs.append(H)
    return shannon_entropies, sampens, apens, Hs
    
    
def custom_features_window(X):
    max_min_dist = max_min_distance(X)
    #energy = energy_sis(X)
    return max_min_dist #, energy

def custom_features_all_data(X):
    max_min_dists = []
    for row in X:
        max_min_dist= custom_features_window(row)
        max_min_dists.append(max_min_dist)
    return max_min_dists


def frequency_domain_features_window(X, fs=250):
    # Compute the power spectral density
    spect, stimes, sfreqs = plot_window_spectrogram(X)
    # Compute the bandpower
    delta, theta, alpha, beta, gamma = compute_bandpower(spect, sfreqs)
    return delta, theta, alpha, beta, gamma

def frequency_domain_features_all_data(X, fs=250):
    deltas = []
    thetas = []
    alphas = []
    betas = []
    gammas = []
    for row in X:
        delta, theta, alpha, beta, gamma = frequency_domain_features_window(row, fs)
        deltas.append(delta)
        thetas.append(theta)
        alphas.append(alpha)
        betas.append(beta)
        gammas.append(gamma)
    return deltas, thetas, alphas, betas, gammas



def plot_predictions_and_signal(
    target,
    data,
    start_time: float,
    stop_time: float,
    data_sampling_rate: int = 250,
    target_duration: int = 2,
    channel_to_plot: int = 0,
):


    # Calculate the start and stop indices for the signal
    start_idx_signal = int(start_time * data_sampling_rate)
    stop_idx_signal = int(stop_time * data_sampling_rate)

    # Calculate the start and stop indices for the labels
    start_idx_label = int(start_time / target_duration)
    stop_idx_label = int(stop_time / target_duration) + 1

    # Slice the data and prediction probabilities
    sliced_signal = data[channel_to_plot, start_idx_signal:stop_idx_signal]
    sliced_prediction_prob = (
        target[channel_to_plot, start_idx_label:stop_idx_label]
    )

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Plot the sliced EEG signal
    ax[0].plot(np.arange(start_idx_signal, stop_idx_signal) / data_sampling_rate, sliced_signal)
    ax[0].set_title("EEG signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")

    ax[1].plot(
        np.arange(start_idx_label, stop_idx_label) * target_duration,
        sliced_prediction_prob,
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Probs")
    ax[1].set_ylim(-0.01, 1.1)
    
    for i in range(len(sliced_prediction_prob)):
        
        if start_time % 2 == 0:
            if i == 0:
                start = start_time
                end = start + 1
            elif i == len(sliced_prediction_prob) - 1:
                start = start_time + (i-1) * 2 + 1
                end = start_time + i * 2
                
            else:
                start = start_time + (i-1) * 2 + 1
                end = start_time + i * 2 + 1
                
        #Not pair not implemented
        else:
            if i == 0:
                start = start_time
                end = start + 2
            else:
                start = start_time + (i-1) * 2 + 1
                end = start_time + (i+1) * 2 + 1 
            
        if sliced_prediction_prob[i] > 0.5:
            color = 'green'
        else:
            color = 'red'
        ax[0].axvspan(start, end, color=color, alpha=0.3)

    if not stop_time - start_time > 50:
        for x in range(start_time, stop_time):
            if x % 2 == 1:
                ax[1].axvline(x=x, color='r', linestyle='--', alpha=0.5)
                ax[0].axvline(x=x, color='r', linestyle='--', alpha=0.5)

    plt.show()
    
    

def reshape_array_into_windows(x, sample_rate, window_duration_in_seconds):
    """
    Reshape the data into an array of shape (C, T, window) where 'window' contains
    the points corresponding to 'window_duration' seconds of data.

    Parameters:
    x (numpy array): The input data array.
    sample_rate (int): The number of samples per second.
    window_duration_in_seconds (float): The duration of each window in seconds.

    Returns:
    reshaped_x (numpy array): The reshaped array with shape (C, T, window).
    """
    # Calculate the number of samples in one window
    window_size = int(window_duration_in_seconds * sample_rate)
    
    # Ensure the total length of x is a multiple of window_size
    total_samples = x.shape[-1]
    if total_samples % window_size != 0:
        # Truncate or pad x to make it divisible by window_size
        x = x[..., :total_samples - (total_samples % window_size)]
    # Reshape x into (C, T, window)
    reshaped_x = x.reshape(x.shape[0], -1, window_size)

    return reshaped_x
   
    
    
def plot_window_spectrogram(X, fs = 250):
    # Set spectrogram params
    fs = fs  # Sampling Frequency
    frequency_range = [0, 60]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = 3  # Set time-half bandwidth
    num_tapers = 5  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    window_params = [1.5, 0.5]  # Window size is 4s with step size of 1s
    min_nfft = 0  # No minimum nfft
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    multiprocess = True  # use multiprocessing
    n_jobs = 3  # use 3 cores in multiprocessing
    weighting = 'unity'  # weight each taper at 1
    plot_on = False  # plot spectrogram
    return_fig = False  # do not return plotted spectrogram
    clim_scale = False # do not auto-scale colormap
    verbose = False  # print extra info
    xyflip = False  # do not transpose spect output matrix


    # Compute the multitaper spectrogram
    spect, stimes, sfreqs = multitaper_spectrogram(X, fs, frequency_range, time_bandwidth, num_tapers, window_params, min_nfft, detrend_opt, multiprocess, n_jobs,
                                                weighting, plot_on, return_fig, clim_scale, verbose, xyflip)
    
    return spect, stimes, sfreqs





def process_data(data):
    filter_cutoff = 25
    fs = 250
    filtered = butter_lowpass_filter(data, cutoff=filter_cutoff, fs=fs, order=3)
    reshaped_data_filtered = reshape_array_into_windows(filtered, sample_rate=fs, window_duration_in_seconds=2)
    reshaped_data_raw = reshape_array_into_windows(data, sample_rate=fs, window_duration_in_seconds=2)
    return reshaped_data_filtered, reshaped_data_raw

def compute_features_on_record(data_raw, data_filtered):
    df = pd.DataFrame()
    max_val, min_val, amplitude, mean, variance, skewness, kurt = time_domain_features_window(data_filtered)
    df['max_val'] = max_val
    df['min_val'] = min_val
    df['amplitude'] = amplitude
    df['mean'] = mean
    df['variance'] = variance
    df['skewness'] = skewness
    df['kurt'] = kurt
    #takes like 30 min to compute
    shannon, sampen, apen, H = complexity_features_all_data(data_filtered)
    df['shannon'] = shannon
    df['sampen'] = sampen
    df['apen'] = apen
    df['H'] = H
    mmd = custom_features_all_data(data_filtered)
    df['mmd'] = mmd
    #takes 1h to compile
    delta, theta, alpha, beta, gamma = frequency_domain_features_all_data(data_raw)
    df['delta'] = delta
    df['theta'] = theta
    df['alpha'] = alpha
    df['beta'] = beta
    df['gamma'] = gamma
    return df

def compute_predictions_on_record (data, model):
    predictions = []
    features = compute_features_on_record(data)
    features = np.array([features[k] for k in features.keys()])
    features = features.swapaxes(0,1).swapaxes(1,2)
    for channel in range(features.shape[0]):
        predictions.append(model.predict(features[channel]))
    return np.array(predictions)

def format_array_to_target_format(array, record_number):
    assert isinstance(record_number, int)
    assert isinstance(array, np.ndarray)
    assert len(array.shape) == 2
    assert array.shape[0] == 5
    assert set(np.unique(array)) == {0, 1}
    formatted_target = []
    for i in range(array.shape[0]):
        channel_encoding = (i + 1) * 100000
        record_number_encoding = record_number * 1000000
        for j in range(array.shape[1]):
            formatted_target.append(
                {
                    "identifier": record_number_encoding + channel_encoding + j,
                    "target": array[i, j],
                }
            )
    return formatted_target

