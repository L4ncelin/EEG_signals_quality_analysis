from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.decomposition import NMF
from sklearn.model_selection import learning_curve
from tqdm import tqdm
import concurrent.futures
import torch

from scipy.stats import skew, kurtosis
from scipy.signal import welch

from Src.features_functions import *

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

def butter_bandpass_filter(data, lowcut=0.1, highcut=50, fs=250, order=4):
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


def segment_signals(data, window_size):
    """
    Segmente les signaux en fenêtres non-chevauchantes.
    
    Args:
        data (numpy.ndarray): Array de dimensions (n_channels, n_samples).
        window_size (int): Nombre de points par fenêtre.
    
    Returns:
        numpy.ndarray: Array de dimensions (n_windows, n_channels, window_size).
    """
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_size  # Nombre de fenêtres possibles
    segmented_data = data[:, :n_windows * window_size]  # Tronquer pour correspondre à des fenêtres entières
    segmented_data = segmented_data.reshape(n_channels, n_windows, window_size)
    return np.transpose(segmented_data, (1, 0, 2))  # Dimensions (n_windows, n_channels, window_size)

def create_segments(signal, segment_length=500, channel:int=0):
    # Initialiser une liste pour stocker les segments
    segments = []

    signal_filtered = butter_bandpass_filter(signal,0.1,30,250,5)

    # Parcourir le signal par pas de 500
    for i in range(0, len(signal_filtered), segment_length):
        # Vérifier si le segment de 500 points peut être extrait
        if i + segment_length <= len(signal_filtered):
            segment = signal_filtered[i:i + segment_length]
            segments.append(segment)

    segments = [arr.tolist() for arr in segments]
    segments_dict = {"signal_windowed_channel{}".format(channel): segments}

    segments_df = pd.DataFrame(segments_dict)
    
    return segments_df

def calculate_segment_speed(window):
    """
    Fonction qui découpe une fenêtre de 500 points en 5 segments de 100 points,
    puis calcule la vitesse moyenne, maximale et minimale pour chaque segment.

    Args:
    - window (array): Une fenêtre de signal EEG de 500 points.

    Returns:
    - (avg_speed, max_speed, min_speed): Vitesse moyenne, maximale et minimale pour les 5 segments.
    """
    segment_length = 100  # Taille de chaque segment
    n_segments = 5  # Nombre de segments (pour 500 points, on aura 5 segments de 100 points)
    
    avg_speeds = []
    max_speeds = []
    min_speeds = []
    
    # Découper la fenêtre en 5 segments
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        segment = window[start_idx:end_idx]
        
        # Calcul de la vitesse sur le segment (différences successives)
        velocity = np.diff(segment)
        
        # Calcul des statistiques de vitesse pour ce segment
        avg_speeds.append(np.mean(velocity))
        max_speeds.append(np.max(velocity))
        min_speeds.append(np.min(velocity))
    
    # Calcul des statistiques globales sur les 5 segments
    avg_speed = np.mean(avg_speeds)
    max_speed = np.max(max_speeds)
    min_speed = np.min(min_speeds)
    
    return avg_speed, max_speed, min_speed

def extract_features(window, other_windows):
    window = np.array(window)

    features = {}
    
    # Indicateurs temporels
    features['std'] = np.std(window, axis=0)
    features['var'] = np.var(window, axis=0)
    features["amplitude"] = np.max(window, axis=0) - np.min(window, axis=0)
    features['skewness'] = skew(window, axis=0)
    features['kurtosis'] = kurtosis(window, axis=0)
    features["min"] = np.min(window, axis=0)
    features["max"] = np.max(window, axis=0)
    features["mean"] = np.mean(window, axis=0)
    features["median"] = np.median(window, axis=0)

    avg_speed, max_speed, min_speed = calculate_segment_speed(window)
    features["amplitude_speed"] = max_speed - min_speed
    features["avg_speed"] = avg_speed
    features["max_speed"] = max_speed
    features["min_speed"] = min_speed
    
    # Énergie
    features['energy'] = np.sum(window**2, axis=0)
    
    # Indicateurs fréquentiels
    freqs, psd = welch(window, fs=250, axis=0, nperseg=window.shape[0])
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    for band, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs < high)
        features[f'psd_{band}'] = np.mean(psd[idx_band], axis=0)

    # Ajout des caractéristiques des ondelettes
    wavelet_features = extract_wavelet_features(window)
    features.update(wavelet_features)

    # Ajout des nouvelles caractéristiques
    features['pfd'] = petrosian_fractal_dimension(window)
    features['hc'] = hjorth_complexity(window)
    features['hm'] = hjorth_mobility(window)
    features['ha'] = hjorth_activity(window)
    features['renyi_entropy'] = renyi_entropy(window)
    features['mcl'] = mean_curve_length(window)
    features['spen'] = spectral_entropy(window)
    features['hurst'] = hurst_exponent(window)
    features['pen'] = permutation_entropy(window)
    features['me'] = mean_energy(window)
    features['apen'] = approximate_entropy(window)
    features['mte'] = mean_teager_energy(window)
    features['zc'] = zero_crossings(window)

    features['wv1'] = np.mean(wigner_ville(window, 1))
    features['wv2'] = np.mean(wigner_ville(window, 2))
    features['wv3'] = np.mean(wigner_ville(window, 3))
    features['wv4'] = np.mean(wigner_ville(window, 4))

    # Calculer les indicateurs pour chaque canal de other_windows
    corr_list = []
    cov_list = []
    dist_list = []

    for other_window in other_windows:
        corr_list.append(np.corrcoef(window, other_window)[0, 1])
        cov_list.append(np.cov(window, other_window)[0, 1])
        dist_list.append(np.linalg.norm(window - other_window))

    # Agréger les indicateurs pour tous les canaux
    features['corr_with_others'] = np.mean(corr_list)  # Moyenne des corrélations
    features['cov_with_others'] = np.mean(cov_list)    # Moyenne des covariances
    features['euclidean_distance_with_others'] = np.mean(dist_list)  # Moyenne des distances euclidiennes
        
    return features

def extract_features2(window, other_windows):
    window = np.array(window)

    features = {}

    # Time Domain Features
    features['std'] = np.std(window, axis=0)
    features['var'] = np.var(window, axis=0)
    features["amplitude"] = np.max(window, axis=0) - np.min(window, axis=0)
    features['skewness'] = skew(window, axis=0)
    features['kurtosis'] = kurtosis(window, axis=0)
    features["min"] = np.min(window, axis=0)
    features["max"] = np.max(window, axis=0)
    features["mean"] = np.mean(window, axis=0)
    features["median"] = np.median(window, axis=0)

    avg_speed, max_speed, min_speed = calculate_segment_speed(window)
    features["amplitude_speed"] = max_speed - min_speed
    features["avg_speed"] = avg_speed
    features["max_speed"] = max_speed
    features["min_speed"] = min_speed
    
    features['energy'] = np.sum(window**2, axis=0)
    
    # Frequency Based Features
    freqs, psd = welch(window, fs=250, axis=0, nperseg=window.shape[0])
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    for band, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs < high)
        features[f'psd_{band}'] = np.mean(psd[idx_band], axis=0)

    # Wigner-Ville Distribution Features
    features['wv1'] = np.mean(wigner_ville(window, 1))
    features['wv2'] = np.mean(wigner_ville(window, 2))
    features['wv3'] = np.mean(wigner_ville(window, 3))
    features['wv4'] = np.mean(wigner_ville(window, 4))

    # Nonlinear Based Features
    features['pfd'] = petrosian_fractal_dimension(window)
    features['hc'] = hjorth_complexity(window)
    features['hm'] = hjorth_mobility(window)
    features['ha'] = hjorth_activity(window)
    features['mcl'] = mean_curve_length(window)
    features['zc'] = zero_crossings(window)
    features['me'] = mean_energy(window)
    
    # Entropy-Based Features
    features['renyi_entropy'] = renyi_entropy(window)
    features['spen'] = spectral_entropy(window)
    features['pen'] = permutation_entropy(window)
    features['apen'] = approximate_entropy(window)
    features['mte'] = mean_teager_energy(window)
    features['hurst'] = hurst_exponent(window)

    # Calculer les indicateurs pour chaque canal de other_windows
    corr_list = []
    cov_list = []
    dist_list = []

    for other_window in other_windows:
        corr_list.append(np.corrcoef(window, other_window)[0, 1])
        cov_list.append(np.cov(window, other_window)[0, 1])
        dist_list.append(np.linalg.norm(window - other_window))

    features['corr_with_others'] = np.mean(corr_list)
    features['cov_with_others'] = np.mean(cov_list)
    features['euclidean_distance_with_others'] = np.mean(dist_list)
        
    return features


# ---------------------- All data without channels infos --------------------- #

# Fonction à paralléliser
def process_window_1(window):
    features = extract_features(window)
    return np.hstack(list(features.values()))

def get_training_input_data():
    data = pd.DataFrame(columns=["signal_windowed", "target"])

    for s in range(4):
        print("Computing set n°{}".format(s))
        train_data, train_target = load_train_data(set=s)

        for c in range(5):
            segments_df = create_segments(train_data[c])

            segments_df["target"] = train_target[c]

            data = pd.concat([data, segments_df])
    
    # Extraire les indicateurs pour chaque fenêtre
    all_features = []
    print("{} windows to compute...".format(data.shape[0]))
    # for window in data["signal_windowed"]:
    #     features = extract_features(window)
    #     all_features.append(np.hstack(list(features.values())))

    features = extract_features(data["signal_windowed"].iloc[0])
    # Appliquer le traitement en parallèle avec concurrent.futures
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_features = list(executor.map(process_window_1, data["signal_windowed"]))

    column_names = []
    for key in features.keys():
        column_names.extend([key])

    df = pd.DataFrame(all_features)

    df = np.clip(df, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    df.columns = column_names

    # Add NMF
    nmf_df = compute_nmf(data, n_components=8)
    df = pd.concat([df, nmf_df], axis=1)


    df["target"] = data["target"].tolist()

    y = df["target"]
    X = df.drop(columns=["target"])

    return X, y, data

def get_testing_input_data():
    data = pd.DataFrame(columns=["signal_windowed"])

    for s in range(2):
        print("Computing set n°{}".format(s))
        test_data = load_test_data(set=s)

        for c in range(5):
            segments_df = create_segments(test_data[c])

            data = pd.concat([data, segments_df])

    
    # Extraire les indicateurs pour chaque fenêtre
    all_features = []
    print("{} windows to compute...".format(data.shape[0]))
    # for window in data["signal_windowed"]:
    #     features = extract_features(window)
    #     all_features.append(np.hstack(list(features.values())))

    features = extract_features(data["signal_windowed"].iloc[0])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_features = list(executor.map(process_window_1, data["signal_windowed"]))

    column_names = []
    for key in features.keys():
        column_names.extend([key])

    df = pd.DataFrame(all_features)

    df = np.clip(df, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    df.columns = column_names

    # Add NMF
    nmf_df = compute_nmf(data, n_components=8)
    df = pd.concat([df, nmf_df], axis=1)

    return df

def compute_nmf(data:pd.DataFrame, n_components:int=8):
    all_windows = np.array(data["signal_windowed"].tolist())
    all_windows = [[x if x >= 0 else 0 for x in arr] for arr in all_windows]

    # Initialiser un modèle NMF
    model = NMF(n_components=n_components, init='random', random_state=42)

    # Ajuster le modèle sur les données
    W = model.fit_transform(all_windows)  # Matrice W (caractéristiques latentes)

    array_df = pd.DataFrame(W, columns=[f"NMF_mode_{i}" for i in range(n_components)])

    return array_df

# --------------------- All features with channels infos --------------------- #
# Fonction pour traiter une fenêtre
def process_window(args):
    window, other_windows = args
    features = extract_features(window, other_windows)
    return np.hstack(list(features.values()))

def get_training_input_data_channel():
    data = pd.DataFrame()

    for c in range(5):
        print("Computing channel n°{}".format(c))

        data_set = pd.DataFrame()
        for s in range(4):
        
            train_data, train_target = load_train_data(set=s)
            segments_df = create_segments(train_data[c], channel=c)

            segments_df["target_channel{}".format(c)] = train_target[c]

            data_set = pd.concat([data_set, segments_df])
        
        data = pd.concat([data, data_set], axis=1)
    
    # Extraire les indicateurs pour chaque fenêtre
    all_features = []
    print("{} windows to compute...".format(data.shape[0]))
    for c in range(5):
        for i in tqdm(range(len(data["signal_windowed_channel0"])), desc="Features extraction for channel n°{}".format(c)):
            window = data["signal_windowed_channel{}".format(c)].iloc[i]
            other_windows = [data["signal_windowed_channel{}".format(j)].iloc[i] for j in range(5) if j != c]
            features = extract_features(window, other_windows)
            all_features.append(np.hstack(list(features.values())))
            

    column_names = []
    for key in features.keys():
        column_names.extend([key])

    df = pd.DataFrame(all_features)

    df = np.clip(df, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    df.columns = column_names

    # Add NMF
    nmf_df = compute_nmf(data, n_components=8)
    df = pd.concat([df, nmf_df], axis=1)


    df["target"] = pd.concat([data["target_channel{}".format(c)] for c in range(5)], axis=0).tolist()

    y = df["target"]
    X = df.drop(columns=["target"])

    return X, y, data

# Fonction principale avec parallélisation
def get_training_input_data_channel_parallel():
    data = pd.DataFrame()

    # Charger les données pour chaque canal
    for c in range(5):
        print("Computing channel n°{}".format(c))

        data_set = pd.DataFrame()
        for s in range(4):
            train_data, train_target = load_train_data(set=s)
            segments_df = create_segments(train_data[c], channel=c)
            segments_df["target_channel{}".format(c)] = train_target[c]
            data_set = pd.concat([data_set, segments_df])

        data = pd.concat([data, data_set], axis=1)

    # Extraction des indicateurs parallélisée
    all_features = []
    print("{} windows to compute...".format(data.shape[0]*5))

    # Créer les arguments pour chaque fenêtre
    tasks = []
    for c in range(5):
        for i in range(len(data["signal_windowed_channel0"])):
            window = data["signal_windowed_channel{}".format(c)].iloc[i]
            other_windows = [data["signal_windowed_channel{}".format(j)].iloc[i] for j in range(5) if j != c]
            tasks.append((window, other_windows))  # Préparer les arguments pour chaque tâche

    # Utiliser ProcessPoolExecutor pour la parallélisation
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_window, tasks),
                total=len(tasks),
                desc="Parallel feature extraction"
            )
        )

    # Convertir les résultats en DataFrame
    all_features = np.array(results)
    column_names = []
    for key in extract_features(np.random.random(500), [np.random.random(500) for _ in range(4)]).keys():
        column_names.append(key)

    df = pd.DataFrame(all_features, columns=column_names)

    # Limiter les valeurs extrêmes
    df = np.clip(df, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    # Ajouter les données NMF
    signal_data = pd.DataFrame(pd.concat([data["signal_windowed_channel{}".format(c)] for c in range(5)], axis=0), columns=["signal_windowed"])
    nmf_df = compute_nmf(signal_data, n_components=8)
    df = pd.concat([df, nmf_df], axis=1)

    # Ajouter la cible
    df["target"] = pd.concat([data["target_channel{}".format(c)] for c in range(5)], axis=0).tolist()

    y = df["target"]
    X = df.drop(columns=["target"])

    return X, y, data

def get_testing_input_data_channel_parallel():
    data = pd.DataFrame()

    # Charger les données pour chaque canal
    for c in range(5):
        print(f"Processing channel n°{c}")
        
        data_set = pd.DataFrame()
        for s in range(2):  # Parcours des deux ensembles de test
            test_data = load_test_data(set=s)
            segments_df = create_segments(test_data[c], channel=c)
            data_set = pd.concat([data_set, segments_df])

        data = pd.concat([data, data_set], axis=1)

    # Extraction des indicateurs parallélisée
    all_features = []
    print(f"{data.shape[0] * 5} windows to compute...")

    # Créer les arguments pour chaque fenêtre
    tasks = []
    for c in range(5):
        for i in range(len(data[f"signal_windowed_channel{c}"])):
            window = data[f"signal_windowed_channel{c}"].iloc[i]
            other_windows = [
                data[f"signal_windowed_channel{j}"].iloc[i] 
                for j in range(5) if j != c
            ]
            tasks.append((window, other_windows))  # Préparer les arguments pour chaque tâche

    # Utiliser ProcessPoolExecutor pour la parallélisation
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_window, tasks),
                total=len(tasks),
                desc="Parallel feature extraction"
            )
        )

    # Convertir les résultats en DataFrame
    all_features = np.array(results)
    column_names = []
    for key in extract_features(np.random.random(500), [np.random.random(500) for _ in range(4)]).keys():
        column_names.append(key)

    df = pd.DataFrame(all_features, columns=column_names)

    # Limiter les valeurs extrêmes
    df = np.clip(df, -np.finfo(np.float32).max, np.finfo(np.float32).max)

    # Ajouter les données NMF
    signal_data = pd.DataFrame(pd.concat([data[f"signal_windowed_channel{c}"] for c in range(5)], axis=0), columns=["signal_windowed"])
    nmf_df = compute_nmf(signal_data, n_components=8)
    df = pd.concat([df, nmf_df], axis=1)

    return df

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=3, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)

    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, val_mean, label="Validation score")
    plt.xlabel("Training size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Learning Curve")
    plt.show()

# ------------------------------------ GPU ----------------------------------- #

# Convertissez les calculs en PyTorch
def extract_features_gpu(window):

    if not torch.cuda.is_available():
        raise RuntimeError("GPU non disponible. Veuillez vérifier votre installation CUDA.")
    
    # Déplacer les données vers le GPU
    window = torch.tensor(window, dtype=torch.float32, device='cuda')

    features = {}
    
    # Indicateurs temporels
    features['std'] = torch.std(window, dim=0).cpu().numpy()
    features["amplitude"] = (torch.max(window, dim=0).values - torch.min(window, dim=0).values).cpu().numpy()
    features['skewness'] = skew(window.cpu().numpy(), axis=0)
    features['kurtosis'] = kurtosis(window.cpu().numpy(), axis=0)

    # Énergie
    features['energy'] = torch.sum(window**2, dim=0).cpu().numpy()

    # Indicateurs fréquentiels
    window_np = window.cpu().numpy()  # Welch nécessite NumPy
    freqs, psd = welch(window_np, fs=250, axis=0, nperseg=window_np.shape[0])
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    for band, (low, high) in bands.items():
        idx_band = (freqs >= low) & (freqs < high)
        features[f'psd_{band}'] = psd[idx_band].mean(axis=0)

    # Ajout des nouvelles caractéristiques
    features['pfd'] = petrosian_fractal_dimension(window.cpu().numpy())
    features['hc'] = hjorth_complexity(window.cpu().numpy())
    features['renyi_entropy'] = renyi_entropy(window.cpu().numpy())
    features['mcl'] = mean_curve_length(window.cpu().numpy())
    features['spen'] = spectral_entropy(window.cpu().numpy())
    features['hurst'] = hurst_exponent(window.cpu().numpy())
    features['pen'] = permutation_entropy(window.cpu().numpy())
    features['me'] = mean_energy(window.cpu().numpy())
    features['apen'] = approximate_entropy(window.cpu().numpy())
    features['mte'] = mean_teager_energy(window.cpu().numpy())
    features['zc'] = zero_crossings(window.cpu().numpy())
    features['wv1'] = torch.mean(torch.tensor(wigner_ville(window.cpu().numpy(), 1))).item()
    features['wv2'] = torch.mean(torch.tensor(wigner_ville(window.cpu().numpy(), 2))).item()
    features['wv3'] = torch.mean(torch.tensor(wigner_ville(window.cpu().numpy(), 3))).item()
    features['wv4'] = torch.mean(torch.tensor(wigner_ville(window.cpu().numpy(), 4))).item()
    features['ha'] = hjorth_activity(window.cpu().numpy())
    features['hm'] = hjorth_mobility(window.cpu().numpy())
    
    return features


# Fonction parallèle modifiée pour GPU
def process_window_gpu(window):
    features = extract_features_gpu(window)
    return np.hstack(list(features.values()))


# Modifiez la fonction principale pour intégrer le GPU
def get_training_input_data_gpu():
    data = pd.DataFrame(columns=["signal_windowed", "target"])

    for s in range(4):
        print("Computing set n°{}".format(s))
        train_data, train_target = load_train_data(set=s)

        for c in range(5):
            segments_df = create_segments(train_data[c])
            segments_df["target"] = train_target[c]
            data = pd.concat([data, segments_df])

    # Extraire les indicateurs pour chaque fenêtre
    all_features = []
    print("{} windows to compute...".format(data.shape[0]))

    # Utiliser concurrent.futures avec GPU
    features = extract_features_gpu(data["signal_windowed"].iloc[0])
    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_features = list(executor.map(process_window_gpu, data["signal_windowed"]))

    column_names = []
    for key in features.keys():
        column_names.extend([key])

    df = pd.DataFrame(all_features, columns=column_names)

    # Ajouter NMF
    nmf_df = compute_nmf(data, n_components=4)
    df = pd.concat([df, nmf_df], axis=1)

    df["target"] = data["target"].tolist()

    y = df["target"]
    X = df.drop(columns=["target"])

    return X, y, data
