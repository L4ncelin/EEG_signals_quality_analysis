from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram


def load_train_data():
    ROOT_PATH = Path("../Data/train/")
    training_data = [(np.load(ROOT_PATH / f"data_{i}.npy"),np.load(ROOT_PATH / f"target_{i}.npy")) for i in range(4)]

    train_data = training_data[1][0]
    train_target = training_data[1][1]

    return train_data, train_target


def plot_predictions_and_signal(
    target,
    data,
    start_time: float,
    stop_time: float,
    data_sampling_rate: int = 250,
    target_duration: int = 2,
    channel_to_plot: int = 0,
):
    """
    
    
    """

    # Calculate the start and stop indices for the signal
    start_idx_signal = int(start_time * data_sampling_rate)
    stop_idx_signal = int(stop_time * data_sampling_rate)

    # Calculate the start and stop indices for the labels
    start_idx_label = int(start_time / target_duration)
    stop_idx_label = int(stop_time / target_duration)

    # Slice the data and prediction probabilities
    sliced_signal = data[channel_to_plot, start_idx_signal:stop_idx_signal]
    sliced_prediction_prob = (
        target[channel_to_plot, start_idx_label:stop_idx_label]
    )

    fig, ax = plt.subplots(2, 1, figsize=(20, 10))

    # Plot the sliced EEG signal
    ax[0].plot(np.arange(len(sliced_signal)) / data_sampling_rate, sliced_signal)
    ax[0].set_title("EEG signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude")

    ax[1].plot(
        np.arange(len(sliced_prediction_prob)) * target_duration,
        sliced_prediction_prob,
    )
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Probs")

    plt.show()


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def plot_signals_analysis(signal:pd.DataFrame, target_signal:pd.DataFrame, fs:int=250, window_size:int=2, duration:int=3600, channel:int=0, start_time:int=0):
    """Plot Spectrogram signal, aligned with real signal and target signal. We can parameter the duration and the start time.

    Args:
        signal (pd.DataFrame): Real signal
        target_signal (pd.DataFrame): Target signal
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        window_size (int, optional): target window size. Defaults to 2 seconds.
        duration (int, optional): duration we want to see. Defaults to 3600 seconds.
        channel (int, optional): channel to use. Defaults to 0.
        start_time (int, optional): starting time. Defaults to 0 seconds.
    """

    # ----------------------------------- Data ----------------------------------- #
    # Indexs
    start_idx_data = int(start_time * fs)
    start_idx_target = int(start_time / window_size)

    end_idx_data = start_idx_data + int(duration * fs)
    end_idx_target = start_idx_target + int(duration / window_size)

    # Signal
    data = signal[channel, start_idx_data:end_idx_data]
    t_data = np.arange(len(data)) / fs

    # Target
    target_data = target_signal[channel, start_idx_target:end_idx_target]
    t_target = np.arange(len(target_data)) * window_size

    # Multitaper
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range=[0,30], plot_on=False)
    multitaper_data = 10 * np.log10(spect)


    # ---------------------------------- Graphs ---------------------------------- #
    fig, axs = plt.subplots(3, 1, figsize=(18,9))

    cax = axs[0].imshow(multitaper_data, aspect="auto", vmin=-6, vmax=30, cmap="jet", extent=[0, t_data[-1], 30, 0])
    axs[0].invert_yaxis()
    axs[0].set_ylabel("Spectrogram")
    axs[0].set_xticklabels([int(i) for i in np.linspace(start_time , duration + start_time, 8)]) # To have the index of the real time we are studying
    #fig.colorbar(cax, ax=axs[0])

    axs[1].plot(t_data, data)
    axs[1].set_xlim(0,t_data[-1])
    axs[1].set_xticklabels([int(i) for i in np.linspace(start_time , duration + start_time, 8)])
    axs[1].set_ylabel("Signal amplitude")

    axs[2].plot(t_target, target_data)
    axs[2].set_xlim(0,t_target[-1])
    axs[2].set_xticklabels([int(i) for i in np.linspace(start_time , duration + start_time, 8)])
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("Defect classification")

    fig.suptitle('Signals analysis for {} seconds'.format(int(t_data[-1])), fontsize=15)

    plt.show()
