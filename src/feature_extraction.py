"""
Moduł ekstrakcji cech z sygnałów EKG.

Cechy używane jako wejście do klasycznych metod uczenia maszynowego:
- Statystyczne (dziedzina czasu): mean, std, min, max, skośność, kurtoza, IQR...
- Energetyczne: RMS, moc
- Spektralne (FFT): centroid, entropia spektralna, moc w pasmach
- Gradientowe (pochodna sygnału): cechy szybkości zmian

Dla sygnału 12-kanałowego (12 odprowadzeń) generowany jest wektor 228 cech
(19 cech × 12 odprowadzeń).
"""

import numpy as np
from scipy import stats, signal as sp_signal
from typing import List


# Nazwy 12 odprowadzeń EKG
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def _time_domain_features(ch_signal: np.ndarray) -> List[float]:
    """
    Cechy dziedziny czasu dla jednego odprowadzenia.

    Obliczane cechy (10 wartości):
    - Wartość średnia (mean)
    - Odchylenie standardowe (std)
    - Wartość minimalna
    - Wartość maksymalna
    - Peak-to-peak (max − min)
    - Skośność (skewness) – asymetria rozkładu amplitudy
    - Kurtoza (kurtosis) – "spiczastość" rozkładu amplitudy
    - 25. percentyl
    - 75. percentyl
    - IQR (Q75 − Q25) – rozstęp ćwiartkowy
    """
    q25, q75 = np.percentile(ch_signal, [25, 75])
    return [
        np.mean(ch_signal),
        np.std(ch_signal),
        np.min(ch_signal),
        np.max(ch_signal),
        np.ptp(ch_signal),           # peak-to-peak
        stats.skew(ch_signal),       # skośność
        stats.kurtosis(ch_signal),   # kurtoza
        q25,
        q75,
        q75 - q25,                   # IQR
    ]


def _energy_features(ch_signal: np.ndarray) -> List[float]:
    """
    Cechy energetyczne dla jednego odprowadzenia.

    Obliczane cechy (3 wartości):
    - Energia sygnału (suma kwadratów)
    - RMS (Root Mean Square) – efektywna wartość
    - Zero-crossing rate – częstość przekroczeń zera (mierzy "aktywność")
    """
    energy = np.sum(ch_signal ** 2)
    rms = np.sqrt(np.mean(ch_signal ** 2))
    zcr = np.sum(np.diff(np.sign(ch_signal)) != 0) / len(ch_signal)
    return [energy, rms, zcr]


def _spectral_features(ch_signal: np.ndarray, fs: int = 100) -> List[float]:
    """
    Cechy spektralne (FFT) dla jednego odprowadzenia.

    Obliczane cechy (4 wartości):
    - Centroid spektralny – "środek ciężkości" widma częstotliwościowego
    - Dominująca częstotliwość – częstotliwość z maksimum widma
    - Moc w paśmie LF [0.5–5 Hz] – niska częstotliwość (fale P, T)
    - Moc w paśmie HF [5–40 Hz] – wysoka częstotliwość (QRS)
    """
    n = len(ch_signal)
    fft_vals = np.abs(np.fft.rfft(ch_signal)) ** 2    # widmo mocy
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    total_power = np.sum(fft_vals) + 1e-10

    # Centroid spektralny (ważona suma częstotliwości)
    spectral_centroid = np.sum(freqs * fft_vals) / total_power

    # Dominująca częstotliwość
    dominant_freq = freqs[np.argmax(fft_vals)]

    # Moc w pasmach diagnostycznych
    lf_mask = (freqs >= 0.5) & (freqs <= 5.0)
    hf_mask = (freqs > 5.0) & (freqs <= 40.0)
    lf_power = np.sum(fft_vals[lf_mask]) / total_power
    hf_power = np.sum(fft_vals[hf_mask]) / total_power

    return [spectral_centroid, dominant_freq, lf_power, hf_power]


def _gradient_features(ch_signal: np.ndarray) -> List[float]:
    """
    Cechy pochodnej sygnału (gradient numeryczny).

    Pochodna sygnału EKG podkreśla szybkie zmiany (kompleksy QRS).
    Obliczane cechy (2 wartości):
    - Średnia wartość bezwzględna gradientu (MAV) – miara aktywności
    - Maksimum wartości bezwzględnej gradientu
    """
    grad = np.diff(ch_signal)
    return [
        np.mean(np.abs(grad)),    # MAV gradientu
        np.max(np.abs(grad)),     # maks. gwałtowność zmiany
    ]


def extract_statistical_features(ecg_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Ekstrakcja wektora cech ze wszystkich 12 odprowadzeń EKG.

    Dla każdego odprowadzenia oblicza:
    - 10 cech czasowych
    - 3 cechy energetyczne
    - 4 cechy spektralne
    - 2 cechy gradientowe
    = 19 cech × 12 odprowadzeń = 228 cech łącznie

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG, kształt (n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania [Hz].

    Zwraca
    ------
    np.ndarray
        Wektor cech (228,).
    """
    features = []

    for ch in range(ecg_signal.shape[1]):
        ch_sig = ecg_signal[:, ch]

        features.extend(_time_domain_features(ch_sig))
        features.extend(_energy_features(ch_sig))
        features.extend(_spectral_features(ch_sig, fs=fs))
        features.extend(_gradient_features(ch_sig))

    return np.array(features, dtype=np.float32)


def get_feature_names() -> List[str]:
    """
    Zwraca listę nazw wszystkich 228 cech (do interpretacji wyników).

    Zwraca
    ------
    list
        Lista stringów z nazwami cech w formacie 'odprowadzenie_cecha'.
    """
    time_names = ['mean', 'std', 'min', 'max', 'ptp', 'skew', 'kurtosis', 'q25', 'q75', 'iqr']
    energy_names = ['energy', 'rms', 'zcr']
    spectral_names = ['spectral_centroid', 'dominant_freq', 'lf_power', 'hf_power']
    gradient_names = ['grad_mav', 'grad_max']

    all_per_lead = time_names + energy_names + spectral_names + gradient_names

    names = []
    for lead in LEAD_NAMES:
        for feat in all_per_lead:
            names.append(f'{lead}_{feat}')
    return names


def extract_features_with_derivatives(ecg_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Ekstrakcja cech z sygnału oryginalnego ORAZ jego pierwszej pochodnej.

    Zgodnie z wymaganiami projektu: analiza wpływu pochodnych sygnału.
    Łączy wektor cech sygnału bazowego (228) z wektorem cech pochodnej (228),
    dając łącznie 456 cech.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania.

    Zwraca
    ------
    np.ndarray
        Połączony wektor cech (456,).
    """
    # Cechy sygnału oryginalnego
    feats_original = extract_statistical_features(ecg_signal, fs=fs)

    # Oblicz pierwszą pochodną
    first_deriv = np.gradient(ecg_signal, axis=0)

    # Cechy pochodnej
    feats_deriv = extract_statistical_features(first_deriv, fs=fs)

    return np.concatenate([feats_original, feats_deriv])


def extract_features_batch(
    X: np.ndarray,
    fs: int = 100,
    use_derivatives: bool = False,
    verbose: bool = True
) -> np.ndarray:
    """
    Ekstrakcja cech dla całego zbioru sygnałów.

    Parametry
    ---------
    X : np.ndarray
        Tablica sygnałów (N, n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania.
    use_derivatives : bool
        Jeśli True – dołącza cechy pochodnej (456 cech zamiast 228).
    verbose : bool
        Czy wyświetlać postęp.

    Zwraca
    ------
    np.ndarray
        Macierz cech (N, n_features).
    """
    features = []
    extractor = extract_features_with_derivatives if use_derivatives else extract_statistical_features

    for i, ecg in enumerate(X):
        if verbose and i % 1000 == 0:
            print(f"  Ekstrakcja cech: {i}/{len(X)}")
        features.append(extractor(ecg, fs=fs))

    return np.array(features, dtype=np.float32)
