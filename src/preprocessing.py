"""
Moduł przetwarzania wstępnego sygnałów EKG.

Zawiera:
- Ocenę jakości sygnału (SNR, clipping, segmenty płaskie)
- Filtrację pasmowoprzepustową (usunięcie dryftu bazowego i szumów HF)
- Filtr notch (eliminacja zakłóceń sieciowych 50 Hz)
- Usunięcie dryftu bazowego (baseline wander)
- Resampling do docelowej częstotliwości
- Normalizację (Z-score i Min-Max)
- Kompletny potok przetwarzania
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import zscore as scipy_zscore
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ocena jakości sygnału
# ─────────────────────────────────────────────────────────────────────────────

def assess_signal_quality(ecg_signal: np.ndarray, fs: int = 100) -> Dict[str, float]:
    """
    Oblicza metryki jakości dla sygnału EKG wielokanałowego.

    Analizowane cechy:
    - SNR (Signal-to-Noise Ratio) – stosunek mocy sygnału do szumu
    - clipping_ratio – udział próbek bliskich nasyceniu przetwornika
    - flat_ratio     – udział okien gdzie sygnał jest stały (zerowe odcinki, brak kontaktu)
    - baseline_drift – maks. odchylenie linii izoelektrycznej w kolejnych oknach

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads) z sygnałem EKG.
    fs : int
        Częstotliwość próbkowania w Hz.

    Zwraca
    ------
    dict
        Słownik z metrykami jakości oraz łącznym `quality_score` ∈ [0, 1].
    """
    metrics = {}
    n_samples, n_leads = ecg_signal.shape

    # ── SNR (przybliżony): moc sygnału / moc szumu wysokoczęstotliwościowego
    signal_power = np.mean(ecg_signal ** 2)
    # Szum ≈ różnica kolejnych próbek (różniczka 1. rzędu odcina składowe wolnozmienne)
    noise_est = np.diff(ecg_signal, axis=0)
    noise_power = np.mean(noise_est ** 2)
    metrics['snr_db'] = 10 * np.log10(signal_power / (noise_power + 1e-10))

    # ── Clipping: sprawdź czy wartości dotykają granic zakresu
    abs_max = np.max(np.abs(ecg_signal))
    if abs_max > 0:
        metrics['clipping_ratio'] = float(
            np.mean(np.abs(ecg_signal) > 0.995 * abs_max)
        )
    else:
        metrics['clipping_ratio'] = 0.0

    # ── Flat ratio: okna o zerowym odchyleniu standardowym (brak sygnału)
    window_size = max(1, fs // 10)        # okno 100 ms
    n_windows = n_samples // window_size
    if n_windows > 0:
        windows = ecg_signal[:n_windows * window_size].reshape(n_windows, window_size, n_leads)
        stds = windows.std(axis=1)        # (n_windows, n_leads)
        metrics['flat_ratio'] = float(np.mean(stds < 1e-6))
    else:
        metrics['flat_ratio'] = 0.0

    # ── Baseline drift: odchylenie mediany w 1-sekundowych oknach
    win = fs
    n_win = n_samples // win
    if n_win >= 2:
        medians = np.array([
            np.median(ecg_signal[i * win:(i + 1) * win], axis=0)
            for i in range(n_win)
        ])
        metrics['baseline_drift'] = float(np.max(np.ptp(medians, axis=0)))
    else:
        metrics['baseline_drift'] = 0.0

    # ── Łączna ocena jakości (heurystyczna, ∈ [0, 1])
    score = 1.0
    if metrics['clipping_ratio'] > 0.01:
        score -= 0.30
    if metrics['flat_ratio'] > 0.05:
        score -= 0.30
    if metrics['snr_db'] < 10:
        score -= 0.20
    if metrics['baseline_drift'] > 0.5:
        score -= 0.10
    metrics['quality_score'] = max(0.0, round(score, 4))

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Filtracja
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: int = 100,
    order: int = 4
) -> np.ndarray:
    """
    Filtr pasmowoprzepustowy Butterwortha.

    Usuwa:
    - Składowe < 0.5 Hz  → dryft bazowy (oddech, ruch)
    - Składowe > 40 Hz   → szumy mięśniowe (EMG) i szumy elektryczne

    Zakres diagnostyczny EKG: 0.05–150 Hz wg AHA;
    dla automatycznej analizy typowo stosuje się 0.5–40 Hz.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).
    lowcut : float
        Dolna częstotliwość odcięcia [Hz].
    highcut : float
        Górna częstotliwość odcięcia [Hz].
    fs : int
        Częstotliwość próbkowania [Hz].
    order : int
        Rząd filtru Butterwortha.

    Zwraca
    ------
    np.ndarray
        Przefiltrowany sygnał, ta sama kształt co wejście.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Projektujemy filtr pasmowoprzepustowy
    b, a = sp_signal.butter(order, [low, high], btype='band')
    # filtfilt: zero-phase filtering (brak przesunięcia fazowego)
    return sp_signal.filtfilt(b, a, ecg_signal, axis=0)


def notch_filter(
    ecg_signal: np.ndarray,
    fs: int = 100,
    freq: float = 50.0,
    Q: float = 30.0
) -> np.ndarray:
    """
    Filtr zaporowy (notch) do usunięcia zakłóceń sieciowych 50 Hz.

    W Polsce (UE) sieć energetyczna pracuje na 50 Hz, co powoduje
    indukcję zakłóceń w elektrodach EKG. Filtr IIR notch eliminuje
    wąskie pasmo wokół tej częstotliwości.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania [Hz].
    freq : float
        Częstotliwość zakłócenia do usunięcia [Hz].
    Q : float
        Czynnik dobroci (szerokość wycięcia: wyższy Q → węższe wycięcie).

    Zwraca
    ------
    np.ndarray
        Sygnał z usuniętym zakłóceniem sieciowym.
    """
    b, a = sp_signal.iirnotch(freq, Q, fs)
    return sp_signal.filtfilt(b, a, ecg_signal, axis=0)


def remove_baseline_wander(ecg_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Usuwa dryft linii bazowej przez odjęcie silnie wygładzonej wersji sygnału.

    Metoda: filtr medianowy dwuprzebiegowy (mediana 200 ms + mediana 600 ms)
    zalecany przez Lyons & Murthy dla zapisu EKG.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania [Hz].

    Zwraca
    ------
    np.ndarray
        Sygnał z usuniętym dryftem bazowym.
    """
    from scipy.ndimage import median_filter

    # Rozmiary okien w próbkach (muszą być nieparzyste)
    w1 = int(0.2 * fs)
    if w1 % 2 == 0:
        w1 += 1
    w2 = int(0.6 * fs)
    if w2 % 2 == 0:
        w2 += 1

    baseline = np.zeros_like(ecg_signal)
    for ch in range(ecg_signal.shape[1]):
        # Dwuprzebiegowy filtr medianowy aproksymuje linię bazową
        baseline[:, ch] = median_filter(
            median_filter(ecg_signal[:, ch], size=w1),
            size=w2
        )
    return ecg_signal - baseline


# ─────────────────────────────────────────────────────────────────────────────
# 3. Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample_signal(
    ecg_signal: np.ndarray,
    orig_fs: int,
    target_fs: int
) -> np.ndarray:
    """
    Dokonuje resampligu sygnału EKG do docelowej częstotliwości próbkowania.

    Używa scipy.signal.resample (FFT-based), który jest dokładny ale wolniejszy
    niż metody interpolacyjne. Dla EKG warto zachować co najmniej 100 Hz.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).
    orig_fs : int
        Oryginalna częstotliwość próbkowania.
    target_fs : int
        Docelowa częstotliwość próbkowania.

    Zwraca
    ------
    np.ndarray
        Resamplowany sygnał.
    """
    if orig_fs == target_fs:
        return ecg_signal
    n_target = int(ecg_signal.shape[0] * target_fs / orig_fs)
    return sp_signal.resample(ecg_signal, n_target, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Normalizacja
# ─────────────────────────────────────────────────────────────────────────────

def normalize_signal(ecg_signal: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalizuje sygnał EKG.

    Dostępne metody:
    - 'zscore'  : standaryzacja do μ=0, σ=1 osobno dla każdego odprowadzenia
    - 'minmax'  : skalowanie do [0, 1] osobno dla każdego odprowadzenia
    - 'robust'  : skalowanie przez medianę i IQR (odporne na wartości odstające)

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).
    method : str
        Metoda normalizacji: 'zscore', 'minmax' lub 'robust'.

    Zwraca
    ------
    np.ndarray
        Znormalizowany sygnał, kształt bez zmian.
    """
    result = np.zeros_like(ecg_signal, dtype=np.float64)

    for ch in range(ecg_signal.shape[1]):
        ch_sig = ecg_signal[:, ch].astype(np.float64)

        if method == 'zscore':
            std = ch_sig.std()
            if std < 1e-10:
                std = 1.0
            result[:, ch] = (ch_sig - ch_sig.mean()) / std

        elif method == 'minmax':
            rng = ch_sig.max() - ch_sig.min()
            if rng < 1e-10:
                rng = 1.0
            result[:, ch] = (ch_sig - ch_sig.min()) / rng

        elif method == 'robust':
            median = np.median(ch_sig)
            iqr = np.percentile(ch_sig, 75) - np.percentile(ch_sig, 25)
            if iqr < 1e-10:
                iqr = 1.0
            result[:, ch] = (ch_sig - median) / iqr

        else:
            raise ValueError(f"Nieznana metoda normalizacji: {method}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Obliczanie pochodnych sygnału
# ─────────────────────────────────────────────────────────────────────────────

def compute_derivatives(ecg_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Oblicza pierwszą i drugą pochodną sygnału EKG (różniczka numeryczna).

    Pochodne sygnału są przydatne w klasyfikacji, ponieważ:
    - 1. pochodna: podkreśla szybkie zmiany (kompleksy QRS, fale P, T)
    - 2. pochodna: wykrywa punkty przegięcia (inflection points)

    Parametry
    ---------
    ecg_signal : np.ndarray
        Macierz (n_samples, n_leads).

    Zwraca
    ------
    Tuple[np.ndarray, np.ndarray]
        (pierwsza_pochodna, druga_pochodna) – ten sam kształt co wejście
        (przez uzupełnienie skrajnych wartości).
    """
    # Centralna różniczka numeryczna (np.gradient automatycznie obsługuje brzegi)
    first_deriv = np.gradient(ecg_signal, axis=0)
    second_deriv = np.gradient(first_deriv, axis=0)
    return first_deriv, second_deriv


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kompletny potok przetwarzania
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pipeline(
    ecg_signal: np.ndarray,
    fs: int = 100,
    target_fs: int = 100,
    normalize_method: str = 'zscore',
    apply_notch: bool = True,
    apply_baseline: bool = True
) -> np.ndarray:
    """
    Kompletny potok przetwarzania wstępnego sygnału EKG.

    Kolejność kroków:
    1. Resampling do docelowej częstotliwości
    2. Usunięcie dryftu bazowego (opcjonalne)
    3. Filtr pasmowoprzepustowy [0.5–40 Hz]
    4. Filtr notch 50 Hz (opcjonalne)
    5. Normalizacja (Z-score domyślnie)

    Parametry
    ---------
    ecg_signal : np.ndarray
        Surowy sygnał EKG, kształt (n_samples, n_leads).
    fs : int
        Oryginalna częstotliwość próbkowania.
    target_fs : int
        Docelowa częstotliwość (po resampligu).
    normalize_method : str
        Metoda normalizacji ('zscore', 'minmax', 'robust').
    apply_notch : bool
        Czy zastosować filtr notch 50 Hz.
    apply_baseline : bool
        Czy usunąć dryft bazowy.

    Zwraca
    ------
    np.ndarray
        Przetworzony sygnał, kształt (n_target_samples, n_leads).
    """
    # Krok 1: Resampling
    ecg = resample_signal(ecg_signal.astype(np.float64), fs, target_fs)

    # Krok 2: Usunięcie dryftu bazowego
    if apply_baseline:
        ecg = remove_baseline_wander(ecg, fs=target_fs)

    # Krok 3: Filtr pasmowoprzepustowy
    ecg = bandpass_filter(ecg, lowcut=0.5, highcut=40.0, fs=target_fs)

    # Krok 4: Filtr notch 50 Hz
    if apply_notch:
        ecg = notch_filter(ecg, fs=target_fs, freq=50.0)

    # Krok 5: Normalizacja
    ecg = normalize_signal(ecg, method=normalize_method)

    return ecg


def preprocess_batch(
    X: np.ndarray,
    fs: int = 100,
    verbose: bool = True,
    **kwargs
) -> np.ndarray:
    """
    Przetwarza cały zbiór sygnałów EKG.

    Parametry
    ---------
    X : np.ndarray
        Tablica (N, n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania.
    verbose : bool
        Czy wyświetlać postęp.
    **kwargs
        Przekazywane do preprocess_pipeline.

    Zwraca
    ------
    np.ndarray
        Przetworzone sygnały, kształt (N, n_target_samples, n_leads).
    """
    processed = []
    for i, ecg in enumerate(X):
        if verbose and i % 1000 == 0:
            print(f"  Przetwarzanie: {i}/{len(X)}")
        processed.append(preprocess_pipeline(ecg, fs=fs, **kwargs))
    return np.array(processed)
