"""
Augmentacja danych dla sygnałów EKG.

Augmentacja polega na tworzeniu zmodyfikowanych kopii istniejących nagrań
w celu sztucznie zwiększenia zbioru treningowego. Dzięki temu model uczy się
być bardziej odporny na naturalne warianty sygnału (ruch pacjenta, szumy,
różnice sprzętowe).

Zaimplementowane techniki:
1. Szum gaussowski (Gaussian Noise)     – symuluje szumy elektryczne elektrody
2. Przesunięcie w czasie (Time Shift)   – symuluje różne punkty startowe nagrania
3. Skalowanie amplitudy (Amplitude)     – kompensuje różnice czułości sprzętu
4. Dryft bazowy (Baseline Wander)       – symuluje ruch pacjenta/oddech
5. Maskowanie odcinka (Time Masking)    – symuluje chwilową utratę kontaktu elektrody

Każda funkcja przyjmuje i zwraca macierz (n_samples, n_leads) – identyczny
format co reszta modułów projektu.
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Techniki augmentacji
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(
    ecg_signal: np.ndarray,
    snr_db: float = 20.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Dodaje szum gaussowski o zadanym stosunku sygnał/szum (SNR).

    W rzeczywistych nagraniach EKG szumy elektryczne elektrody i wzmacniacza
    mogą mieć różne poziomy. Augmentacja szumem uczy model rozpoznawać wzorce
    pomimo zaszumionego sygnału.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads). Powinien być znormalizowany.
    snr_db : float
        Docelowy stosunek sygnał/szum w dB. Typowe wartości: 15-30 dB.
        Im niższe SNR, tym więcej szumu.
    rng : np.random.Generator, opcjonalne
        Generator losowy (dla reprodukowalności).

    Zwraca
    ------
    np.ndarray
        Sygnał z dodanym szumem, ten sam kształt.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Moc sygnału
    signal_power = np.mean(ecg_signal ** 2)
    # Oblicz wymaganą moc szumu: SNR = 10*log10(P_signal/P_noise)
    noise_power  = signal_power / (10 ** (snr_db / 10))
    # Generuj biały szum gaussowski o odpowiedniej amplitudzie
    noise = rng.normal(0, np.sqrt(noise_power), ecg_signal.shape)

    return ecg_signal + noise


def time_shift(
    ecg_signal: np.ndarray,
    max_shift_fraction: float = 0.1,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Przesuwa sygnał o losową liczbę próbek wzdłuż osi czasu.

    Różne systemy EKG mogą startować nagranie w różnym miejscu cyklu pracy
    serca. Augmentacja przesunięciem uczy model rozpoznawać diagnozę niezależnie
    od fazy cyklu.

    Metoda: np.roll (przesunięcie cykliczne – koniec sygnału trafia na początek).

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    max_shift_fraction : float
        Maksymalne przesunięcie jako ułamek długości sygnału (domyślnie 10%).
    rng : np.random.Generator, opcjonalne

    Zwraca
    ------
    np.ndarray
        Przesunięty sygnał, ten sam kształt.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = ecg_signal.shape[0]
    max_shift = int(n_samples * max_shift_fraction)
    # Losowe przesunięcie: ujemne = w lewo, dodatnie = w prawo
    shift = rng.integers(-max_shift, max_shift + 1)

    return np.roll(ecg_signal, shift, axis=0)


def scale_amplitude(
    ecg_signal: np.ndarray,
    scale_range: tuple = (0.8, 1.2),
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Skaluje amplitudę sygnału przez losowy współczynnik.

    Różne urządzenia EKG mogą mieć różną czułość przetwornika.
    Skalowanie uczy model rozpoznawać wzorce morfologiczne niezależnie
    od bezwzględnej amplitudy sygnału.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    scale_range : tuple
        Zakres współczynnika skalowania (min, max). Domyślnie ±20%.
    rng : np.random.Generator, opcjonalne

    Zwraca
    ------
    np.ndarray
        Przeskalowany sygnał, ten sam kształt.
    """
    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(scale_range[0], scale_range[1])
    return ecg_signal * scale


def add_baseline_wander(
    ecg_signal: np.ndarray,
    fs: int = 100,
    amplitude: float = 0.1,
    freq_range: tuple = (0.05, 0.5),
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Dodaje syntetyczny dryft bazowy (sinusoidalny).

    Dryft bazowy w EKG pochodzi z oddechu (0.1-0.5 Hz) i ruchów pacjenta.
    Jest to jeden z najczęstszych artefaktów w zapisach EKG.
    Augmentacja uczy model ignorowania powolnych zmian linii izoelektrycznej.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    fs : int
        Częstotliwość próbkowania [Hz].
    amplitude : float
        Maksymalna amplituda dryftu (w jednostkach znormalizowanego sygnału).
    freq_range : tuple
        Zakres częstotliwości dryftu [Hz]. Domyślnie 0.05–0.5 Hz (oddech).
    rng : np.random.Generator, opcjonalne

    Zwraca
    ------
    np.ndarray
        Sygnał z dryftem bazowym, ten sam kształt.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples, n_leads = ecg_signal.shape
    t = np.linspace(0, n_samples / fs, n_samples)
    result = ecg_signal.copy()

    for ch in range(n_leads):
        # Losowa częstotliwość i faza dla każdego odprowadzenia
        freq  = rng.uniform(freq_range[0], freq_range[1])
        phase = rng.uniform(0, 2 * np.pi)
        amp   = rng.uniform(amplitude * 0.3, amplitude)
        wander = amp * np.sin(2 * np.pi * freq * t + phase)
        result[:, ch] += wander

    return result


def time_masking(
    ecg_signal: np.ndarray,
    max_mask_fraction: float = 0.05,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Zeruje losowy ciągły odcinek sygnału (maskowanie).

    Symuluje chwilową utratę kontaktu elektrody lub artefakt ruchowy.
    Uczy model klasyfikować EKG nawet przy brakujących fragmentach.

    Metoda: wypełnienie wybranego okna wartością 0 (linia izoelektryczna).

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    max_mask_fraction : float
        Maksymalna długość maski jako ułamek długości sygnału (domyślnie 5%).
    rng : np.random.Generator, opcjonalne

    Zwraca
    ------
    np.ndarray
        Sygnał z zamaskowanym odcinkiem, ten sam kształt.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = ecg_signal.shape[0]
    max_len   = int(n_samples * max_mask_fraction)
    mask_len  = rng.integers(1, max(2, max_len + 1))
    start     = rng.integers(0, n_samples - mask_len)

    result = ecg_signal.copy()
    result[start:start + mask_len, :] = 0.0   # zerowanie = linia izoelektryczna

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Zbiorczy potok augmentacji
# ─────────────────────────────────────────────────────────────────────────────

# Nazwy technik (do pętli i wykresów)
AUGMENTATION_NAMES = [
    'Szum gaussowski',
    'Przesunięcie czasowe',
    'Skalowanie amplitudy',
    'Dryft bazowy',
    'Maskowanie odcinka',
]


def apply_single_augmentation(
    ecg_signal: np.ndarray,
    technique: str,
    fs: int = 100,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Stosuje jedną wybraną technikę augmentacji do sygnału.

    Parametry
    ---------
    ecg_signal : np.ndarray
        Sygnał EKG (n_samples, n_leads).
    technique : str
        Jedna z: 'noise', 'shift', 'scale', 'wander', 'mask'.
    fs : int
        Częstotliwość próbkowania.
    rng : np.random.Generator, opcjonalne

    Zwraca
    ------
    np.ndarray
        Augmentowany sygnał.
    """
    if rng is None:
        rng = np.random.default_rng()

    funcs = {
        'noise':  lambda x: add_gaussian_noise(x, snr_db=20.0, rng=rng),
        'shift':  lambda x: time_shift(x, max_shift_fraction=0.1, rng=rng),
        'scale':  lambda x: scale_amplitude(x, scale_range=(0.8, 1.2), rng=rng),
        'wander': lambda x: add_baseline_wander(x, fs=fs, amplitude=0.15, rng=rng),
        'mask':   lambda x: time_masking(x, max_mask_fraction=0.05, rng=rng),
    }
    if technique not in funcs:
        raise ValueError(f"Nieznana technika: {technique}. Dostępne: {list(funcs)}")

    return funcs[technique](ecg_signal)


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    techniques: list = None,
    augment_factor: int = 1,
    fs: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """
    Tworzy powiększony zbiór treningowy przez augmentację każdej próbki.

    Dla każdej próbki losowo wybiera jedną technikę augmentacji i dodaje
    nową, zmodyfikowaną kopię do zbioru. Oryginalne dane są zachowane.

    Parametry
    ---------
    X : np.ndarray
        Oryginalne sygnały (N, n_samples, n_leads).
    y : np.ndarray
        Etykiety (N,).
    techniques : list, opcjonalne
        Lista technik do zastosowania. Domyślnie wszystkie 5.
    augment_factor : int
        Ile dodatkowych kopii per próbka (1 = podwojenie zbioru).
    fs : int
        Częstotliwość próbkowania.
    seed : int
        Ziarno losowe dla reprodukowalności.
    verbose : bool
        Czy drukować postęp.

    Zwraca
    ------
    tuple (X_aug, y_aug)
        Powiększone zbiory danych (oryginalne + augmentowane).
    """
    if techniques is None:
        techniques = ['noise', 'shift', 'scale', 'wander', 'mask']

    rng = np.random.default_rng(seed)
    X_new, y_new = [], []

    for i, (x, label) in enumerate(zip(X, y)):
        if verbose and i % 500 == 0:
            print(f"  Augmentacja: {i}/{len(X)}")

        for _ in range(augment_factor):
            # Losowo wybierz jedną technikę dla tej próbki
            tech = rng.choice(techniques)
            x_aug = apply_single_augmentation(x, tech, fs=fs, rng=rng)
            X_new.append(x_aug)
            y_new.append(label)

    X_aug = np.concatenate([X, np.array(X_new)], axis=0)
    y_aug = np.concatenate([y, np.array(y_new)], axis=0)

    if verbose:
        print(f"\nRozmiar po augmentacji: {len(X)} → {len(X_aug)} próbek")

    return X_aug, y_aug
