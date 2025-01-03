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

def load_test_data(set:int=0):
    """Load all testing sets.

    Args:
        set (int, optional): Testing set in [0,1]. Defaults to 0.

    Returns:
        dict: _description_
    """
    ROOT_PATH = Path("../Data/test/")
    test_data = [np.load(ROOT_PATH / f"data_{i}.npy") for i in [4,5]]

    test_data = test_data[set]

    return test_data


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

def get_multitaper(signal:pd.DataFrame, fs:int=250, duration:int=3600, channel:int=0, start_time:int=0, window_size:list=[5,1]):
    """Return the multitaper matrix.

    Args:
        signal (pd.DataFrame): Real signal
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        duration (int, optional): Duration we want to see. Defaults to 3600 seconds.
        channel (int, optional): Channel to use. Defaults to 0.
        start_time (int, optional): Starting time. Defaults to 0 seconds.
        window_size (list, optional): Window of computing analysis of the multitaper algorithm (precision). Defaults to [5, 1] (len (seconds), step(seconds)).

    Returns:
        np.ndarray : Multitaper matrix as np.array
    """
    # Indexs
    start_idx_data = int(start_time * fs)
    end_idx_data = start_idx_data + int(duration * fs)

    # Signal
    data = signal[channel, start_idx_data:end_idx_data]

    # Multitaper
    try :
        spect, _, _ = multitaper_spectrogram(data, fs, frequency_range=[0,30], plot_on=False, verbose=False, window_params=window_size)
        multitaper_data = 10 * np.log10(spect)
    except ValueError:
        multitaper_data = np.array([])

    return multitaper_data

def get_all_multitapers_data(fs:int=250, nb_sets:int=4, nb_channels:int=5):
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

def process_channel_all_multitapers(args):
    """Parralel task, getting the multitaper matrix of one set of one channel.

    Args:
        args (_type_): arguments of the task function

    Returns:
        np.ndarray: multitaper data
    """
    train_data, fs, duration, c, idx = args
    result = get_multitaper(train_data, fs, duration, channel=c)
    return idx, result

def parallel_all_multitapers(nb_sets:int=4, nb_channels:int=5, fs:int=250):
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
        futures = {executor.submit(process_channel_all_multitapers, task): task for task in tasks}

        # Affichage de la barre de progression
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Calcul multitaper"):
            idx, result = future.result()
            multitaper_data[idx] = result  # Placer le résultat au bon endroit dans la liste

    return multitaper_data


def get_windowed_multitaper_data(fs:int=250, nb_sets:int=4, nb_channels:int=5, duration:int=2, type:str="train", window_size:list=[2,1]):
    """Return all multitapers matrix with a certain windows restriction.

    Args:
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        nb_sets (int, optional): Number of sets. Defaults to 4.
        nb_channels (int, optional): Number of channels. Defaults to 5.
        duration (int, optional): Duration of the windows to capture. Defaults to 2 seconds.
        window_size (list, optional): Window of computing analysis of the multitaper algorithm (precision). Defaults to [2, 1] (len(seconds), step(seconds)).

    Returns:
        list[list[np.ndarray]]: All set and channel multitapers data windowed 
    """
    multitaper_data = []
    x=1
    for i in range(nb_sets):
        set_data = []

        if type == "train":
            train_data, _ = load_train_data(set=i)
        elif type == "test":
            train_data = load_test_data(set=i)

        for c in range(nb_channels):
            channel_data = []
            print("Iteration {}/{}".format(x, nb_sets*nb_channels))
            x+=1

            for w in range(0, train_data.shape[1]//fs , duration): # We windowed the data to get multitaper of 2 seconds
                multitaper = get_multitaper(train_data, fs, duration=duration, channel=c, start_time=w, window_size=window_size)

                if multitaper.size == 0:
                    print("Last window removed for channel {} in set {}".format(c, i))
                else:
                    channel_data += [multitaper]
            
            set_data += [channel_data]
        
        multitaper_data += [set_data]
    
    return multitaper_data

def get_channeled_multitaper_data(fs:int=250, nb_sets:int=4, nb_channels:int=5, duration:int=2, type:str="train", window_size:list=[1,0.5]):
    """Return all multitapers images (31x3) for each channels and sets.

    Args:
        fs (int, optional): Sampling rate. Defaults to 250Hz.
        nb_sets (int, optional): Number of sets. Defaults to 4.
        nb_channels (int, optional): Number of channels. Defaults to 5.
        duration (int, optional): Duration of the windows to capture. Defaults to 2 seconds.
        window_size (list, optional): Window of computing analysis of the multitaper algorithm (precision). Defaults to [1, 0.5] (len(seconds), step(seconds)).

    Returns:
        list[list[5*np.ndarray]]: All set and channel multitapers data windowed 
    """
    multitaper_data = []
    x=1
    for s in range(nb_sets):
        set_data = []

        if type == "train":
            train_data, _ = load_train_data(set=s)
        elif type == "test":
            train_data = load_test_data(set=s)
        
        print("Set {}/{}".format(s, nb_sets))
        x+=1

        for w in range(0, train_data.shape[1]//fs , duration): # We windowed the data to get multitaper images of 2 seconds
            window_data = []
            
            bad_window_flag = False

            for c in range(nb_channels):
            
                multitaper = get_multitaper(train_data, fs, duration=duration, channel=c, start_time=w, window_size=window_size)

                if multitaper.size == 0 or multitaper.shape[1] != 5:
                    print("window {} removed in set {}".format(w, s))
                    x+=2
                    bad_window_flag = True
                    break
                else:
                    window_data += [multitaper]
            
            if bad_window_flag == False:
                set_data += [window_data]
            else:
                bad_window_flag = False
        
        multitaper_data += [set_data]
    
    return multitaper_data


def plot_signals_analysis(signal:pd.DataFrame, target_signal:pd.DataFrame, fs:int=250, window_size:list=[2,1], duration:int=3600, channel:int=0, start_time:int=0, set:int=0, window_target:int=2):
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
    start_idx_target = int(start_time / window_target)

    end_idx_data = start_idx_data + int(duration * fs)
    end_idx_target = start_idx_target + int(duration / window_target)

    # Signal
    data = signal[channel, start_idx_data:end_idx_data]
    t_data = np.arange(len(data)) / fs

    # Target
    target_data = target_signal[channel, start_idx_target:end_idx_target]
    t_target = np.arange(len(target_data)) * window_target

    # Multitaper
    if duration > 5 :
        duration = 5 # To manage small windows, else ignore to plot the entire signal
    spect, stimes, sfreqs = multitaper_spectrogram(data, fs, frequency_range=[0,30], plot_on=False, window_params=window_size)
    multitaper_data = 10 * np.log10(spect)


    # ---------------------------------- Graphs ---------------------------------- #
    fig, axs = plt.subplots(3, 1, figsize=(18,9))
    
    cax = axs[0].imshow(multitaper_data, aspect="auto", vmin=-6, vmax=30, cmap="jet", extent=[0, t_data[-1], 30, 0])
    axs[0].invert_yaxis()
    axs[0].set_ylabel("Frequencies (Hz)")

    axs[1].plot(t_data, data)
    axs[1].set_xlim(0,t_data[-1])
    axs[1].set_ylabel("Signal amplitude")

    axs[2].step(t_target, target_data)
    axs[2].set_xlim(0,t_target[-1])
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("Defect classification")

    cbar = fig.colorbar(cax, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Power (dB)', rotation=270, labelpad=20)

    fig.suptitle('Signals analysis for {} seconds in channel n°{} of training set n°{}'.format(int(t_data[-1]), channel, set), fontsize=15)

    plt.show()

    return spect, stimes, sfreqs

def format_array_to_target_format(array, record_number, nb_points):

    formatted_target = []
    for i in range(5):
        channel_encoding = (i + 1) * 100000
        record_number_encoding = record_number * 1000000
        for j in range(nb_points):
            formatted_target.append(
                {
                    "identifier": record_number_encoding + channel_encoding + j,
                    "target": array[i][j],
                }
            )
    return formatted_target


def create_submission_file(test_data_model, model, output_file_name, conversion:bool=False, channels:bool=False):
    results = []

    # Set 4
    X_test_4 = test_data_model[:66020]
    preds = (model.predict(X_test_4) > 0.5)
    sublists = [preds[i:i + 13204] for i in range(0, len(preds), 13204)]

    formatted_preds = format_array_to_target_format(sublists, 4, 13204)
    if channels is True:
        formatted_preds = formatted_preds.flatten().tolist()
    results.extend(formatted_preds)

    # Set 5
    X_test_5 = test_data_model[66020:]
    preds = (model.predict(X_test_5) > 0.5)
    sublists = [preds[i:i + 9319] for i in range(0, len(preds), 9319)]

    formatted_preds = format_array_to_target_format(sublists, 5, 9319)
    if channels is True:
        formatted_preds = formatted_preds.flatten().tolist()
    results.extend(formatted_preds)

    df = pd.DataFrame(results)

    if conversion is True:
        df['target'] = df['target'].apply(lambda x: 1 if x[0] == True else (0 if x[0] == False else x[0]))
    
    print(df["target"].count(0))

    df.to_csv("../Results/{}.csv".format(output_file_name),index = False)