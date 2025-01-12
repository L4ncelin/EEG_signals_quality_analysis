import numpy as np
from hurst import compute_Hc
from pyentrp import entropy as ent
from scipy.signal import spectrogram
from scipy.signal import welch
import pywt

def petrosian_fractal_dimension(x):
    N = len(x)
    diff = np.diff(x)
    count = np.sum(np.abs(diff) > np.mean(np.abs(diff)))
    return np.log(N) / (np.log(N) + np.log(count / (N - 1)))

def hjorth_complexity(window):
    d1 = np.diff(window)  # Première dérivée
    d2 = np.diff(d1)     # Deuxième dérivée
    activity = np.var(window)
    mobility = np.sqrt(np.var(d1) / activity)
    complexity = np.sqrt(np.var(d2) / np.var(d1)) / mobility
    return complexity

def renyi_entropy(window, alpha=2):
    prob_density = np.histogram(window, bins=10, density=True)[0]
    prob_density = prob_density[prob_density > 0]  # Eviter les valeurs nulles
    return 1 / (1 - alpha) * np.log(np.sum(prob_density**alpha))

def mean_curve_length(window):
    return np.mean(np.abs(np.diff(window)))

def spectral_entropy(window):
    freqs, psd = welch(window, fs=250, axis=0, nperseg=window.shape[0])
    psd = psd / np.sum(psd)  # Normaliser la PSD
    return -np.sum(psd * np.log2(psd))

def hurst_exponent(window):
    H, c, data_reg = compute_Hc(window)
    return H

def permutation_entropy(window, m=3, tau=1):
    return ent.permutation_entropy(window, m, tau)

def mean_energy(window):
    return np.mean(window**2)

def mean_teager_energy(window):
    return np.mean(window**2 - np.roll(window, 1) * np.roll(window, -1))

def zero_crossings(window):
    return np.count_nonzero(np.diff(np.sign(window)))

def wigner_ville(window, order=1):
    _, _, Zxx = spectrogram(window, fs=250)
    return np.abs(Zxx) ** order  # Par exemple, WV-1, WV-2, etc.

def hjorth_activity(x):
    # Calculer la variance du signal
    variance_x = np.var(x)

    # Calculer la dérivée du signal
    dx = np.diff(x)

    # Calculer la variance de la dérivée
    variance_dx = np.var(dx)

    # La Hjorth Activity est la variance de la dérivée divisée par la variance du signal
    activity = variance_dx / variance_x
    
    return activity

def hjorth_mobility(window):
    d1 = np.diff(window)
    return np.sqrt(np.var(d1) / np.var(window))

def approximate_entropy(U, m=2, r=0.2):
    """
    Calculer l'Approximate Entropy (ApEn) d'une série temporelle.
    
    Parameters:
        U (list ou np.array): Série temporelle.
        m (int): Longueur des motifs à comparer.
        r (float): Seuil d'acceptation (généralement 0.2 * std(U)).
    
    Returns:
        float: Valeur de l'Approximate Entropy.
    """
    U = np.array(U)
    N = len(U)
    
    def _phi(m):
        X = np.array([U[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) / (N - m + 1.0)
        return np.sum(np.log(C)) / (N - m + 1.0)
    
    return abs(_phi(m) - _phi(m + 1))

# Fonction pour extraire les coefficients d'ondelettes et les caractéristiques associées
def extract_wavelet_features(window, wavelet='db4', level=5):
    # Calcul de la transformée en ondelettes discrète (DWT)
    coeffs = pywt.wavedec(window, wavelet, level=level)
    
    features = {}
    
    # Boucle sur chaque sous-bande de détail D3 à D5 et approximation A5
    for i in range(3, 6):  # D3 à D5 (D3 = D3-1, D4 = D4-1, D5 = D5-1)
        D = coeffs[i]
        
        # D3-1, D4-1, D5-1 : Moyenne absolue des coefficients
        features[f'D{i}-1'] = np.mean(np.abs(D))
        
        # D3-2, D4-2, D5-2 : Moyenne de la puissance des coefficients
        features[f'D{i}-2'] = np.mean(D**2)
        
        # D3-3, D4-3, D5-3 : Écart-type des coefficients
        features[f'D{i}-3'] = np.std(D)
        
        # D3-4, D4-4, D5-4 : Rapport des moyennes absolues des sous-bandes adjacentes
        if i < 5:
            D_next = coeffs[i+1]
            features[f'D{i}-4'] = np.mean(np.abs(D)) / np.mean(np.abs(D_next))
    
    # A5 : Approximation finale
    A5 = coeffs[0]
    
    # A5-1 : Moyenne absolue des coefficients d'approximation
    features['A5-1'] = np.mean(np.abs(A5))
    
    # A5-2 : Moyenne de la puissance des coefficients d'approximation
    features['A5-2'] = np.mean(A5**2)
    
    # A5-3 : Écart-type des coefficients d'approximation
    features['A5-3'] = np.std(A5)
    
    return features

