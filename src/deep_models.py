"""
Architektury głębokich sieci neuronowych do klasyfikacji sygnałów EKG.

Zaimplementowane architektury (≥3 zgodnie z wymaganiami projektu):
1. CNN1D       – prosta sieć konwolucyjna 1D
2. ResNet1D    – głęboka sieć rezydualna dla sygnałów 1D (Rajpurkar et al.)
3. BiLSTM      – dwukierunkowa sieć LSTM

Każda architektura przyjmuje dane w formacie (batch, n_leads, n_samples)
i zwraca logity (batch, n_classes).

Moduł zawiera również:
- ECGDataset    – Dataset PyTorch dla sygnałów EKG
- Trainer       – klasa pomocnicza do trenowania i ewaluacji modeli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    """
    Dataset PyTorch dla sygnałów EKG.

    Konwertuje sygnały EKG (n_samples, n_leads) do formatu wymaganego przez
    sieci konwolucyjne 1D i LSTM: (n_leads, n_samples).

    Parametry
    ---------
    X : np.ndarray
        Tablica sygnałów (N, n_samples, n_leads).
    y : np.ndarray
        Etykiety klas (N,) jako liczby całkowite.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Transponujemy: (N, n_samples, n_leads) → (N, n_leads, n_samples)
        # taki format wymagany przez Conv1d(in_channels=n_leads)
        self.X = torch.FloatTensor(X).permute(0, 2, 1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Architektura 1: CNN1D
# ─────────────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    Prosta sieć konwolucyjna 1D do klasyfikacji sygnałów EKG.

    Architektura: 3 bloki konwolucyjne + 2 warstwy FC.
    Każdy blok: Conv1d → BatchNorm → ReLU → MaxPool

    BatchNorm po każdej konwolucji stabilizuje trening (przyspiesza zbieżność
    i zmniejsza wrażliwość na learning rate).
    Dropout przed warstwą wyjściową zapobiega overfittingowi.

    Parametry
    ---------
    num_channels : int
        Liczba odprowadzeń EKG (domyślnie 12).
    num_classes : int
        Liczba klas wyjściowych (domyślnie 5 superklasowych).
    """

    def __init__(self, num_channels: int = 12, num_classes: int = 5):
        super(CNN1D, self).__init__()

        # Blok 1: wyodrębnia lokalne cechy niskiej częstotliwości
        self.block1 = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)
        )

        # Blok 2: głębsze cechy o średniej granularności
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)
        )

        # Blok 3: abstrakcyjne cechy wysokiego poziomu
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.2)
        )

        # Adaptacyjny pooling → stały rozmiar niezależnie od długości wejścia
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)

        # Warstwy w pełni połączone (klasyfikator)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametry
        ---------
        x : torch.Tensor
            Kształt (batch, n_leads, n_samples).

        Zwraca
        ------
        torch.Tensor
            Logity, kształt (batch, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = x.flatten(start_dim=1)    # (batch, 128*8)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Architektura 2: ResNet1D
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    """
    Blok rezydualny dla sygnałów 1D (He et al., 2016).

    Połączenie skip-connection pozwala na trening bardzo głębokich sieci
    przez ominięcie problemu zanikającego gradientu. Jeśli f(x) ≈ 0,
    blok uczy się "nic nie robić" (identity mapping).

    Schemat: x → [Conv → BN → ReLU → Conv → BN] + x → ReLU

    Parametry
    ---------
    in_channels : int
        Liczba kanałów wejściowych.
    out_channels : int
        Liczba kanałów wyjściowych.
    stride : int
        Stride konwolucji (>1 zmniejsza rozdzielczość czasową).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock1D, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=7, stride=stride, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        # Projekcja skip-connection gdy wymiary się zmieniają
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity         # residual connection
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    ResNet1D – sieć rezydualna dla jednowymiarowych sygnałów EKG.

    Inspirowana: Hannun et al. "Cardiologist-level arrhythmia detection with
    a mobile ECG device" (2019) oraz He et al. "Deep Residual Learning for
    Image Recognition" (2016) adaptowana do sygnałów 1D.

    Architektura:
    - Warstwa wejściowa: Conv(12→64, k=15) + BN + ReLU + MaxPool
    - 4 grupy bloków rezydualnych: 64→64, 64→128, 128→256, 256→512
    - Global Average Pooling
    - Warstwa klasyfikacyjna (FC)

    Parametry
    ---------
    num_channels : int
        Liczba odprowadzeń (12 dla EKG).
    num_classes : int
        Liczba klas wyjściowych (5 superklasowych).
    """

    def __init__(self, num_channels: int = 12, num_classes: int = 5):
        super(ResNet1D, self).__init__()

        # Warstwa wejściowa – duże jądro, żeby wychwycić cykl serca (~1s przy 100Hz = 100 próbek)
        self.input_block = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Grupy bloków rezydualnych
        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, n_blocks=2, stride=2)

        # Global Average Pooling – redukuje wymiar czasowy do 1
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Klasyfikator
        self.fc = nn.Linear(512, num_classes)

        # Inicjalizacja wag (He initialization dla ReLU)
        self._initialize_weights()

    def _make_layer(self, in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        """Tworzy grupę bloków rezydualnych."""
        blocks = [ResidualBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        """Inicjalizacja He (kaiming) – optymalna dla aktywacji ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametry
        ---------
        x : torch.Tensor
            Kształt (batch, n_leads, n_samples).

        Zwraca
        ------
        torch.Tensor
            Logity, kształt (batch, num_classes).
        """
        x = self.input_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)   # (batch, 512, 1)
        x = x.squeeze(-1)              # (batch, 512)
        x = self.fc(x)                 # (batch, num_classes)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Architektura 3: Bidirectional LSTM
# ─────────────────────────────────────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Dwukierunkowa sieć LSTM do klasyfikacji sekwencji EKG.

    LSTM (Long Short-Term Memory) jest siecią rekurencyjną zdolną do
    modelowania długoterminowych zależności w danych sekwencyjnych.
    Wersja dwukierunkowa (bidirectional) przetwarza sygnał zarówno
    od przodu do tyłu (forward) jak i od tyłu do przodu (backward),
    co pozwala uwzględnić kontekst z obu kierunków czasowych.

    Parametry
    ---------
    input_size : int
        Liczba kanałów/cech na każdą próbkę czasu (= n_leads = 12).
    hidden_size : int
        Liczba jednostek w warstwie ukrytej LSTM.
    num_layers : int
        Liczba warstw LSTM.
    num_classes : int
        Liczba klas wyjściowych.
    dropout : float
        Współczynnik dropout między warstwami LSTM.
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Warstwa normalizacji wejścia
        self.input_norm = nn.LayerNorm(input_size)

        # Dwukierunkowy LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # wejście: (batch, seq_len, features)
            bidirectional=True,         # forward + backward
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention – ważona agregacja sekwencji ukrytych stanów
        self.attention = nn.Linear(hidden_size * 2, 1)

        # Klasyfikator (×2 bo bidirectional)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametry
        ---------
        x : torch.Tensor
            Kształt (batch, n_leads, n_samples) – format z ECGDataset.

        Zwraca
        ------
        torch.Tensor
            Logity, kształt (batch, num_classes).
        """
        # Transpozycja: (batch, n_leads, n_samples) → (batch, n_samples, n_leads)
        # LSTM oczekuje (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)           # (batch, seq_len, n_leads)
        x = self.input_norm(x)

        # Przejście przez LSTM
        # out: (batch, seq_len, hidden_size*2) – wszystkie stany ukryte
        out, _ = self.lstm(x)

        # Mechanizm uwagi (attention): uczymy się ważyć ważność każdej chwili czasu
        attn_weights = torch.softmax(self.attention(out), dim=1)  # (batch, seq_len, 1)
        context = (out * attn_weights).sum(dim=1)                  # (batch, hidden_size*2)

        return self.classifier(context)


# ─────────────────────────────────────────────────────────────────────────────
# Klasa trenera (Trainer)
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Klasa pomocnicza do trenowania i ewaluacji modeli PyTorch.

    Parametry
    ---------
    model : nn.Module
        Model do trenowania.
    device : str
        'cuda', 'mps' lub 'cpu'.
    learning_rate : float
        Współczynnik uczenia dla optymalizatora Adam.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        learning_rate: float = 1e-3
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def train_epoch(self, loader: DataLoader) -> float:
        """Trenuje model przez jedną epokę, zwraca średnią stratę."""
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping – zapobiega eksplozji gradientu (ważne dla LSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item() * len(y_batch)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        """Ewaluuje model, zwraca (strata, dokładność)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)

            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        return total_loss / total, correct / total

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        verbose: bool = True
    ) -> dict:
        """
        Pętla treningowa.

        Parametry
        ---------
        train_loader : DataLoader
            DataLoader z danymi treningowymi.
        val_loader : DataLoader
            DataLoader z danymi walidacyjnymi.
        epochs : int
            Liczba epok.
        verbose : bool
            Czy drukować postęp.

        Zwraca
        ------
        dict
            Historia treningowa (straty i dokładność).
        """
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            epoch_time = time.time() - t0

            self.scheduler.step(val_loss)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"  Epoka [{epoch:3d}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | "
                      f"Czas: {epoch_time:.1f}s")

        # Przywróć najlepsze wagi
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)

        return self.history

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generuje predykcje i prawdziwe etykiety dla całego DataLoadera.

        Zwraca
        ------
        Tuple[np.ndarray, np.ndarray]
            (y_true, y_pred)
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            logits = self.model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

        return np.array(all_labels), np.array(all_preds)


def get_device() -> str:
    """
    Zwraca najlepsze dostępne urządzenie do obliczeń PyTorch.

    Kolejność preferencji: CUDA (GPU NVIDIA) → MPS (Apple Silicon) → CPU
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_deep_models(num_channels: int = 12, num_classes: int = 5) -> dict:
    """
    Zwraca słownik wszystkich zaimplementowanych architektur DL.

    Parametry
    ---------
    num_channels : int
        Liczba odprowadzeń EKG.
    num_classes : int
        Liczba klas wyjściowych.

    Zwraca
    ------
    dict
        {nazwa_modelu: instancja_modelu}
    """
    return {
        'CNN1D': CNN1D(num_channels=num_channels, num_classes=num_classes),
        'ResNet1D': ResNet1D(num_channels=num_channels, num_classes=num_classes),
        'BiLSTM': BiLSTMClassifier(input_size=num_channels, num_classes=num_classes),
    }
