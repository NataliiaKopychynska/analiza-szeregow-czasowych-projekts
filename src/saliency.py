"""
Mapy istotności (Saliency Maps) dla modeli głębokiego uczenia na sygnałach EKG.

Interpretowalność modeli DL polega na odpowiedzi na pytanie:
"Które fragmenty sygnału wejściowego najbardziej wpłynęły na decyzję modelu?"

Zaimplementowane metody:
1. Vanilla Gradient Saliency  – ∂output/∂input (najprostsze, najszybsze)
2. Gradient × Input           – gradient pomnożony przez wartość wejścia
                                  (uwzględnia amplitudę sygnału)
3. Uwaga Attention (BiLSTM)   – bezpośrednia wizualizacja wag mechanizmu uwagi

Wszystkie metody zwracają macierz istotności w formacie (n_leads, n_samples),
zgodnym z formatem wejścia sieci (po transponowaniu przez ECGDataset).

Literatura:
- Simonyan et al. "Deep Inside Convolutional Networks" (2013)
- Shrikumar et al. "Learning Important Features Through Propagating Activation
  Differences" (DeepLIFT, 2017)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vanilla Gradient Saliency
# ─────────────────────────────────────────────────────────────────────────────

def vanilla_saliency(
    model: nn.Module,
    X_sample: np.ndarray,
    target_class: int,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Oblicza mapę istotności przez gradient wyjścia względem wejścia (Vanilla Saliency).

    Metoda: ∂score_c / ∂x  gdzie c = klasa docelowa, x = sygnał wejściowy.

    Intuicja: jeśli mała zmiana próbki x_i powoduje dużą zmianę wyniku klasy c,
    to próbka x_i jest "ważna" dla tej predykcji.

    Zalety: prosta, szybka, model-agnostyczna.
    Wady: może być szumna (gradient jest lokalny, nie globalny).

    Parametry
    ---------
    model : nn.Module
        Wytrenowany model PyTorch (CNN1D lub ResNet1D).
    X_sample : np.ndarray
        Pojedynczy sygnał EKG (n_samples, n_leads) – przed transponowaniem.
    target_class : int
        Indeks klasy dla której liczymy gradient.
    device : str
        Urządzenie PyTorch.

    Zwraca
    ------
    np.ndarray
        Mapa istotności (n_leads, n_samples) – wartości bezwzględne gradientu.
    """
    model.eval()
    model.to(device)

    # Konwersja: (n_samples, n_leads) → (1, n_leads, n_samples) – format sieci
    x_tensor = torch.FloatTensor(X_sample).permute(1, 0).unsqueeze(0)  # (1, n_leads, n_samples)
    x_tensor = x_tensor.to(device)
    x_tensor.requires_grad_(True)

    # Przejście w przód i obliczenie gradientu
    logits = model(x_tensor)                         # (1, n_classes)
    score  = logits[0, target_class]                 # skalar
    model.zero_grad()
    score.backward()

    # Gradient: (1, n_leads, n_samples) → (n_leads, n_samples)
    saliency = x_tensor.grad.detach().cpu().numpy()[0]
    return np.abs(saliency)   # wartość bezwzględna = "ważność" niezależna od kierunku


# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradient × Input
# ─────────────────────────────────────────────────────────────────────────────

def gradient_x_input(
    model: nn.Module,
    X_sample: np.ndarray,
    target_class: int,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Oblicza mapę istotności Gradient × Input.

    Metoda: |∂score_c/∂x · x|  (gradient pomnożony przez amplitudę wejścia).

    Ulepszona wersja Vanilla Saliency – uwzględnia nie tylko czułość modelu
    na daną próbkę, ale też jej rzeczywistą wartość.
    Przykład: gradient może być duży w miejscu płaskiego sygnału (małe x),
    ale iloczyn ∂/∂x · x będzie mały, bo amplituda jest nieistotna.

    Parametry
    ---------
    model : nn.Module
        Wytrenowany model PyTorch.
    X_sample : np.ndarray
        Pojedynczy sygnał EKG (n_samples, n_leads).
    target_class : int
        Indeks klasy docelowej.
    device : str

    Zwraca
    ------
    np.ndarray
        Mapa istotności (n_leads, n_samples).
    """
    model.eval()
    model.to(device)

    x_tensor = torch.FloatTensor(X_sample).permute(1, 0).unsqueeze(0)
    x_tensor = x_tensor.to(device)
    x_tensor.requires_grad_(True)

    logits = model(x_tensor)
    score  = logits[0, target_class]
    model.zero_grad()
    score.backward()

    grad     = x_tensor.grad.detach().cpu().numpy()[0]    # (n_leads, n_samples)
    inp      = x_tensor.detach().cpu().numpy()[0]         # (n_leads, n_samples)

    # Iloczyn punkt-po-punkcie, wartość bezwzględna
    return np.abs(grad * inp)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Wagi uwagi (Attention Weights) z BiLSTM
# ─────────────────────────────────────────────────────────────────────────────

def get_attention_weights(
    model: nn.Module,
    X_sample: np.ndarray,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Wyciąga wagi mechanizmu uwagi (attention) z modelu BiLSTM.

    Mechanizm uwagi uczy się przypisywać każdej chwili czasu wagę:
    - wysoka waga = model skupia się na tym fragmencie sygnału
    - niska waga  = ten fragment jest ignorowany przy podejmowaniu decyzji

    Model attention w BiLSTMClassifier:
        attn_weights = softmax(Linear(lstm_out))  → (batch, seq_len, 1)

    Parametry
    ---------
    model : nn.Module
        Wytrenowany model BiLSTMClassifier.
    X_sample : np.ndarray
        Pojedynczy sygnał EKG (n_samples, n_leads).
    device : str

    Zwraca
    ------
    np.ndarray
        Wagi uwagi (n_samples,) – suma = 1 (rozkład prawdopodobieństwa nad czasem).
    """
    model.eval()
    model.to(device)

    # Format: (1, n_leads, n_samples) → BiLSTM permutuje wewnętrznie na (1, n_samples, n_leads)
    x_tensor = torch.FloatTensor(X_sample).permute(1, 0).unsqueeze(0).to(device)

    # Uruchamiamy forward hook żeby przechwycić wagi attention
    attention_weights = {}

    def hook_fn(module, input, output):
        # output to (batch, seq_len, 1) – wagi softmax
        attention_weights['weights'] = output.detach().cpu().numpy()

    # Zarejestruj hook na warstwie attention
    hook = model.attention.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(x_tensor)

    hook.remove()

    # Wyciągnij wagi: (1, seq_len, 1) → (seq_len,)
    # Uwaga: to są wagi PRZED softmax – aplikujemy softmax manualnie
    raw_weights = attention_weights['weights'][0, :, 0]   # (seq_len,)
    attn = np.exp(raw_weights) / np.sum(np.exp(raw_weights))  # softmax
    return attn


# ─────────────────────────────────────────────────────────────────────────────
# Funkcje pomocnicze – batch i agregacja
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_saliency(
    model: nn.Module,
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    method: str = 'gradient_x_input',
    device: str = 'cpu',
    n_classes: int = 5,
) -> dict:
    """
    Oblicza uśrednione mapy istotności dla każdej klasy diagnostycznej.

    Dla każdej klasy bierze wszystkie próbki tej klasy, oblicza indywidualne
    mapy istotności, a następnie uśrednia po próbkach.
    Wynikowa mapa pokazuje: "które regiony sygnału są typowo ważne dla klasy X?"

    Parametry
    ---------
    model : nn.Module
        Wytrenowany model CNN1D lub ResNet1D.
    X_batch : np.ndarray
        Zbiór sygnałów (N, n_samples, n_leads).
    y_batch : np.ndarray
        Etykiety (N,) – liczby całkowite 0..n_classes-1.
    method : str
        'vanilla' lub 'gradient_x_input'.
    device : str
    n_classes : int

    Zwraca
    ------
    dict
        {class_idx: np.ndarray (n_leads, n_samples)} – uśrednione mapy per klasa.
    """
    saliency_fn = vanilla_saliency if method == 'vanilla' else gradient_x_input
    class_maps = {c: [] for c in range(n_classes)}

    for x, y in zip(X_batch, y_batch):
        # Obliczamy mapę dla faktycznej klasy próbki
        sal = saliency_fn(model, x, int(y), device=device)
        class_maps[int(y)].append(sal)

    # Uśrednianie
    avg_maps = {}
    for c in range(n_classes):
        if class_maps[c]:
            avg_maps[c] = np.mean(class_maps[c], axis=0)
        else:
            avg_maps[c] = None

    return avg_maps


def lead_importance(saliency_map: np.ndarray) -> np.ndarray:
    """
    Redukuje mapę istotności (n_leads, n_samples) do ważności per odprowadzenie.

    Metoda: suma wartości bezwzględnych wzdłuż osi czasu → (n_leads,).
    Odzwierciedla ogólną rolę każdego z 12 odprowadzeń EKG w klasyfikacji.

    Parametry
    ---------
    saliency_map : np.ndarray
        Mapa istotności (n_leads, n_samples).

    Zwraca
    ------
    np.ndarray
        Wektor ważności odprowadzeń (n_leads,), znormalizowany do [0, 1].
    """
    importance = np.sum(saliency_map, axis=1)   # suma po czasie
    if importance.max() > 0:
        importance = importance / importance.max()
    return importance
