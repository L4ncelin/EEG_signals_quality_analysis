from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram


def load_train_data(set:int=0):
    """Load training set 

    Args:
        set (int, optional): Training set in [0,1,2,3]. Defaults to 0.

    Returns:
        _type_: _description_
    """
    ROOT_PATH = Path("../Data/train/")
    training_data = [(np.load(ROOT_PATH / f"data_{i}.npy"),np.load(ROOT_PATH / f"target_{i}.npy")) for i in range(4)]

    train_data = training_data[set][0]
    train_target = training_data[set][1]

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

def get_multitaper(signal:pd.DataFrame, fs:int=250, duration:int=3600, channel:int=0, start_time:int=0):
    """Return the multitaper matrix.

    Args:
        signal (pd.DataFrame): Real signal
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        duration (int, optional): Duration we want to see. Defaults to 3600 seconds.
        channel (int, optional): Channel to use. Defaults to 0.
        start_time (int, optional): Starting time. Defaults to 0 seconds.

    Returns:
        np.ndarray : Multitaper matrix as np.array
    """
    # Indexs
    start_idx_data = int(start_time * fs)
    end_idx_data = start_idx_data + int(duration * fs)

    # Signal
    data = signal[channel, start_idx_data:end_idx_data]

    # Multitaper
    spect, _, _ = multitaper_spectrogram(data, fs, frequency_range=[0,30], plot_on=False, verbose=False)
    multitaper_data = 10 * np.log10(spect)

    return multitaper_data

def get_multitaper_data(fs:int=250, nb_sets:int=4, nb_channels:int=5):
    """Return all multitapers matrix.

    Args:
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        nb_sets (int, optional): Number of sets. Defaults to 4.
        nb_channels (int, optional): Number of channels. Defaults to 5.

    Returns:
        list[np.ndarray]: Multitapers
    """
    multitaper_data = []
    x=1
    for i in range(nb_sets):
        train_data, _ = load_train_data(set=i)
        for c in range(nb_channels):
            print("Iteration {}/{}".format(x, nb_sets*nb_channels))
            x+=1
            multitaper_data += [get_multitaper(train_data, fs, duration=int(train_data.shape[1])/fs, channel=c)]
    
    return multitaper_data

def process_channel(args):
    """Parralel task, getting the multitaper matrix of one set of one channel.

    Args:
        args (_type_): arguments of the task function

    Returns:
        np.ndarray: multitaper data
    """
    train_data, fs, duration, c, idx = args
    result = get_multitaper(train_data, fs, duration, channel=c)
    return idx, result

def parallel_multitaper(nb_sets:int=4, nb_channels:int=5, fs:int=250):
    """Return all multitapers matrix using parralel computation.

    Args:
        nb_sets (int, optional): Number of sets. Defaults to 4.
        nb_channels (int, optional): Number of channels. Defaults to 5.
        fs (int, optional): Sampling rate. Defaults to 250Hz.

    Returns:
        list[np.ndarray]: Multitapers
    """
    tasks = []
    index = 0

    for i in range(nb_sets):
        train_data, _ = load_train_data(set=i)
        duration = int(train_data.shape[1]) / fs
        for c in range(nb_channels):
            tasks.append((train_data, fs, duration, c, index))  # Préparer les arguments pour chaque tâche
            index += 1

    multitaper_data = [None] * len(tasks)  # Créer une liste vide de la bonne taille

    # Exécution parallèle avec barre de progression
    with ProcessPoolExecutor() as executor:
        # Soumettre toutes les tâches
        futures = {executor.submit(process_channel, task): task for task in tasks}

        # Affichage de la barre de progression
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calcul multitaper"):
            idx, result = future.result()
            multitaper_data[idx] = result  # Placer le résultat au bon endroit dans la liste

    return multitaper_data



def plot_signals_analysis(signal:pd.DataFrame, target_signal:pd.DataFrame, fs:int=250, window_size:int=2, duration:int=3600, channel:int=0, start_time:int=0, set:int=0):
    """Plot Spectrogram signal, aligned with real signal and target signal. We can parameter the duration and the start time.

    Args:
        signal (pd.DataFrame): Real signal
        target_signal (pd.DataFrame): Target signal
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        window_size (int, optional): Target window size. Defaults to 2 seconds.
        duration (int, optional): Duration we want to see. Defaults to 3600 seconds.
        channel (int, optional): Channel to use. Defaults to 0.
        start_time (int, optional): Starting time. Defaults to 0 seconds.
        set (int,optional): Training set. Defaults to 0.
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
    axs[0].set_ylabel("Frequencies (Hz)")

    axs[1].plot(t_data, data)
    axs[1].set_xlim(0,t_data[-1])
    axs[1].set_ylabel("Signal amplitude")

    axs[2].plot(t_target, target_data)
    axs[2].set_xlim(0,t_target[-1])
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("Defect classification")

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)

    fig.suptitle('Signals analysis for {} seconds in channel n°{} of training set n°{}'.format(int(t_data[-1]), channel, set), fontsize=15)

    plt.show()

    return spect, stimes, sfreqs
