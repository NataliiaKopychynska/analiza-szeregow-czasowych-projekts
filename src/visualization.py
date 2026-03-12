"""
Moduł wizualizacji dla projektu klasyfikacji EKG.

Zawiera gotowe funkcje do generowania wykresów:
- Sygnały EKG (12 odprowadzeń, porównanie klas)
- Efekty przetwarzania wstępnego (kroki potoku, widma)
- Ocena jakości sygnałów
- Wyniki klasyfikacji (macierze pomyłek, porównanie modeli)
- Krzywe uczenia (Deep Learning)
- Ważność cech (Feature Importance)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
from typing import Optional, List, Dict

# ── Paleta kolorów projektu ──────────────────────────────────────────────────
CLASS_COLORS = {
    'NORM': '#2196F3',   # niebieski
    'MI':   '#F44336',   # czerwony
    'STTC': '#FF9800',   # pomarańczowy
    'CD':   '#4CAF50',   # zielony
    'HYP':  '#9C27B0',   # fioletowy
}
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4',  'V5',  'V6']

# ── Styl globalny ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 120,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.grid': True,
    'grid.alpha': 0.3,
})
sns.set_theme(style='whitegrid', palette='tab10')


# ─────────────────────────────────────────────────────────────────────────────
# 1. Wizualizacje sygnałów EKG
# ─────────────────────────────────────────────────────────────────────────────

def plot_ecg_12lead(
    signal: np.ndarray,
    fs: int = 100,
    title: str = '',
    color: str = 'steelblue',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Rysuje pełny 12-odprowadzeniowy zapis EKG w układzie 4×3.

    Parametry
    ---------
    signal : np.ndarray
        Sygnał EKG (n_samples, 12).
    fs : int
        Częstotliwość próbkowania [Hz].
    title : str
        Tytuł wykresu.
    color : str
        Kolor sygnału.
    save_path : str, optional
        Ścieżka zapisu PNG.
    """
    t = np.arange(signal.shape[0]) / fs
    fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

    for idx, (ax, name) in enumerate(zip(axes.flatten(), LEAD_NAMES)):
        ax.plot(t, signal[:, idx], color=color, linewidth=0.9, alpha=0.9)
        ax.set_title(f'Odprowadzenie {name}', fontsize=10)
        ax.set_ylabel('mV', fontsize=8)
        if idx >= 9:
            ax.set_xlabel('Czas [s]', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_ecg_comparison(
    signals: Dict[str, np.ndarray],
    lead_idx: int = 1,
    fs: int = 100,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Porównuje przykładowe EKG z każdej klasy diagnostycznej na jednym wykresie.

    Parametry
    ---------
    signals : dict
        {nazwa_klasy: sygnał_ndarray (n_samples, 12)}.
    lead_idx : int
        Indeks odprowadzenia do wyświetlenia (domyślnie II = 1).
    """
    n = len(signals)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(list(signals.values())[0].shape[0]) / fs

    for ax, (cls_name, sig) in zip(axes, signals.items()):
        color = CLASS_COLORS.get(cls_name, 'steelblue')
        ax.plot(t, sig[:, lead_idx], color=color, linewidth=0.9)
        ax.set_ylabel(f'mV', fontsize=9)
        ax.set_title(
            f'Klasa: {cls_name} – odprowadzenie {LEAD_NAMES[lead_idx]}',
            color=color, fontsize=11
        )

    axes[-1].set_xlabel('Czas [s]')
    fig.suptitle('Porównanie przykładowych sygnałów EKG wg klasy diagnostycznej',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Wizualizacje przetwarzania wstępnego
# ─────────────────────────────────────────────────────────────────────────────

def plot_preprocessing_steps(
    ecg_raw: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Wizualizuje wszystkie kroki potoku przetwarzania na jednym wykresie.
    Importuje preprocessing z src – brak kołowego importu.
    """
    from src.preprocessing import (
        remove_baseline_wander, bandpass_filter,
        notch_filter, normalize_signal,
    )

    t = np.arange(ecg_raw.shape[0]) / fs
    lead_name = LEAD_NAMES[lead_idx]

    steps = [
        ('Sygnał surowy',                    ecg_raw,                                        '#607D8B'),
        ('Usunięcie dryftu bazowego',         remove_baseline_wander(ecg_raw, fs=fs),         '#4CAF50'),
        ('Filtr pasmowoprzepustowy [0.5–40 Hz]',
            bandpass_filter(remove_baseline_wander(ecg_raw, fs), fs=fs),                       '#FF9800'),
        ('Filtr notch 50 Hz',
            notch_filter(bandpass_filter(remove_baseline_wander(ecg_raw, fs), fs), fs=fs),    '#9C27B0'),
        ('Normalizacja Z-score',
            normalize_signal(
                notch_filter(bandpass_filter(
                    remove_baseline_wander(ecg_raw, fs), fs), fs), 'zscore'),                  '#F44336'),
    ]

    fig, axes = plt.subplots(len(steps), 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f'Potok przetwarzania wstępnego – odprowadzenie {lead_name}',
                 fontsize=13, fontweight='bold')

    for ax, (label, sig, color) in zip(axes, steps):
        ax.plot(t, sig[:, lead_idx], color=color, linewidth=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Amplituda')

    axes[-1].set_xlabel('Czas [s]')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_spectra_comparison(
    ecg_raw: np.ndarray,
    ecg_filtered: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Porównanie widm mocy sygnału przed i po filtracji.
    Wyraźnie pokazuje usunięcie zakłóceń 50 Hz i dryftu poniżej 0.5 Hz.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Widmo mocy sygnału EKG przed i po filtracji',
                 fontsize=13, fontweight='bold')

    for ax, (sig, label, color) in zip(axes, [
        (ecg_raw[:, lead_idx],      'Sygnał surowy',    '#607D8B'),
        (ecg_filtered[:, lead_idx], 'Po filtracji',     '#F44336'),
    ]):
        n = len(sig)
        freqs = rfftfreq(n, d=1.0 / fs)
        power = np.abs(rfft(sig)) ** 2

        ax.semilogy(freqs, power, color=color, linewidth=0.9, alpha=0.85)
        ax.axvspan(0, 0.5,  alpha=0.08, color='red',    label='Usuwany dryft (<0.5 Hz)')
        ax.axvspan(40, 50,  alpha=0.08, color='orange', label='Usuwane szumy (>40 Hz)')
        ax.axvline(50, color='red', linestyle='--', linewidth=1.2, label='50 Hz (sieć)')
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Częstotliwość [Hz]')
        ax.set_ylabel('Moc [log]')
        ax.set_xlim(0, fs / 2)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_normalization_comparison(
    ecg_filtered: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Porównanie trzech metod normalizacji (Z-score, Min-Max, Robust)."""
    from src.preprocessing import normalize_signal

    t = np.arange(ecg_filtered.shape[0]) / fs
    methods = [
        ('zscore', 'Z-score  (μ=0, σ=1)',        '#F44336'),
        ('minmax', 'Min-Max  ([0, 1])',            '#4CAF50'),
        ('robust', 'Robust   (mediana/IQR)',       '#9C27B0'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Porównanie metod normalizacji – odprowadzenie {LEAD_NAMES[lead_idx]}',
                 fontsize=13, fontweight='bold')

    for ax, (method, label, color) in zip(axes, methods):
        normed = normalize_signal(ecg_filtered, method=method)
        ch = normed[:, lead_idx]
        ax.plot(t, ch, color=color, linewidth=0.85)
        stats = f'min={ch.min():.2f}  max={ch.max():.2f}  μ={ch.mean():.4f}  σ={ch.std():.4f}'
        ax.set_title(f'{label}   [{stats}]', fontsize=10)
        ax.set_ylabel('Amplituda')

    axes[-1].set_xlabel('Czas [s]')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_derivatives(
    ecg_processed: np.ndarray,
    fs: int = 100,
    lead_idx: int = 1,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wizualizuje sygnał oraz jego 1. i 2. pochodną."""
    from src.preprocessing import compute_derivatives

    t = np.arange(ecg_processed.shape[0]) / fs
    d1, d2 = compute_derivatives(ecg_processed)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Sygnał EKG i jego pochodne – odprowadzenie {LEAD_NAMES[lead_idx]}',
                 fontsize=13, fontweight='bold')

    for ax, sig, label, color in zip(axes,
        [ecg_processed[:, lead_idx], d1[:, lead_idx], d2[:, lead_idx]],
        ['Sygnał przetworzony (x)',
         '1. pochodna (dx/dt) – podkreśla kompleksy QRS',
         '2. pochodna (d²x/dt²) – punkty przegięcia'],
        ['steelblue', '#FF9800', '#4CAF50']
    ):
        ax.plot(t, sig, color=color, linewidth=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Amplituda')
    axes[-1].set_xlabel('Czas [s]')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_quality_dashboard(
    quality_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Dashboard oceny jakości sygnałów – 4 histogramy metryk jakości.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Dashboard oceny jakości sygnałów EKG', fontsize=13, fontweight='bold')

    metrics = [
        ('snr_db',        'SNR [dB]',               'Stosunek sygnału do szumu'),
        ('clipping_ratio','Clipping Ratio',           'Udział nasyconych próbek'),
        ('flat_ratio',    'Flat Ratio',               'Udział płaskich segmentów'),
        ('quality_score', 'Quality Score  [0–1]',    'Łączna ocena jakości'),
    ]

    cmap = plt.cm.RdYlGn
    for ax, (col, xlabel, title) in zip(axes.flatten(), metrics):
        vals = quality_df[col].dropna()
        median = vals.median()
        # Kolor słupków wg mediany (zielony = dobry)
        if col == 'quality_score':
            bar_color = cmap(median)
        elif col == 'snr_db':
            bar_color = cmap(min(1.0, median / 30))
        else:
            bar_color = cmap(1.0 - min(1.0, median * 10))

        ax.hist(vals, bins=25, color=bar_color, edgecolor='white', alpha=0.85)
        ax.axvline(median, color='navy', linestyle='--', linewidth=1.5,
                   label=f'Mediana: {median:.3f}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Liczba sygnałów')
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Wizualizacje wyników klasyfikacji
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrices(
    results: dict,
    class_names: List[str],
    title: str = '',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Rysuje macierze pomyłek dla wielu modeli na jednym wykresie.

    Parametry
    ---------
    results : dict
        {nazwa_modelu: {'confusion_matrix': ndarray, 'accuracy': float, 'f1_macro': float}}
    class_names : list
        Nazwy klas.
    """
    n = len(results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(title, fontsize=13, fontweight='bold')

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i // ncols][i % ncols]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=res['confusion_matrix'],
            display_labels=class_names
        )
        disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
        acc_str = f"Acc={res.get('accuracy', 0):.3f}"
        f1_str  = f"F1={res.get('f1_macro', 0):.3f}"
        ax.set_title(f'{name}\n{acc_str}  |  {f1_str}', fontsize=10)

    # Ukryj puste komórki
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_model_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Wykres grupowany porównujący Accuracy, F1 Macro, F1 Weighted dla wszystkich modeli.

    Parametry
    ---------
    summary_df : pd.DataFrame
        DataFrame z kolumnami: Model, Accuracy, F1 Macro, F1 Weighted.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Porównanie modeli klasyfikacji EKG', fontsize=13, fontweight='bold')

    models = summary_df['Model'].tolist()
    x = np.arange(len(models))
    w = 0.30

    # F1-score
    axes[0].bar(x - w/2, summary_df['F1 Macro'],    w, label='F1 Macro',    color='steelblue', alpha=0.88)
    axes[0].bar(x + w/2, summary_df['F1 Weighted'], w, label='F1 Weighted', color='coral',     alpha=0.88)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=28, ha='right')
    axes[0].set_ylabel('F1-score')
    axes[0].set_title('F1-score (Macro vs Weighted)')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    for bar in axes[0].patches:
        h = bar.get_height()
        if h > 0.01:
            axes[0].text(bar.get_x() + bar.get_width()/2, h + 0.01,
                         f'{h:.2f}', ha='center', va='bottom', fontsize=8)

    # Accuracy vs Balanced Accuracy
    if 'Balanced Accuracy' in summary_df.columns:
        axes[1].bar(x - w/2, summary_df['Accuracy'],          w, label='Accuracy',          color='seagreen',  alpha=0.88)
        axes[1].bar(x + w/2, summary_df['Balanced Accuracy'], w, label='Balanced Accuracy', color='darkorange', alpha=0.88)
    else:
        axes[1].bar(x, summary_df['Accuracy'], w * 1.5, label='Accuracy', color='seagreen', alpha=0.88)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=28, ha='right')
    axes[1].set_ylabel('Dokładność')
    axes[1].set_title('Accuracy')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 25,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Poziomy wykres słupkowy top-N najważniejszych cech (np. z Random Forest).
    Kolory odpowiadają odprowadzeniu EKG.
    """
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = importances[top_idx]

    # Kolor wg odprowadzenia (prefiks nazwy cechy)
    lead_colors = plt.cm.tab20(np.linspace(0, 1, 12))
    def lead_color(feat_name):
        for j, lead in enumerate(LEAD_NAMES):
            if feat_name.startswith(lead + '_'):
                return lead_colors[j]
        return 'steelblue'

    colors = [lead_color(n) for n in top_names]

    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel('Ważność cechy (Gini impurity)')
    ax.set_title(f'Top {top_n} najważniejszych cech – Random Forest', fontweight='bold')

    # Legenda odprowadzeń
    patches = [mpatches.Patch(color=lead_colors[j], label=lead)
               for j, lead in enumerate(LEAD_NAMES)]
    ax.legend(handles=patches, title='Odprowadzenie', ncol=2, fontsize=8,
              loc='lower right')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Wizualizacje Deep Learning
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    histories: Dict[str, dict],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Rysuje krzywe uczenia (loss i accuracy) dla wielu modeli DL.

    Parametry
    ---------
    histories : dict
        {nazwa_modelu: {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Krzywe uczenia – Głębokie sieci neuronowe', fontsize=13, fontweight='bold')

    colors_map = {'CNN1D': 'steelblue', 'ResNet1D': 'darkorange', 'BiLSTM': 'green'}

    for name, hist in histories.items():
        color = colors_map.get(name, 'gray')
        epochs = range(1, len(hist['train_loss']) + 1)

        axes[0].plot(epochs, hist['train_loss'], '--', color=color, alpha=0.6,
                     linewidth=1.2, label=f'{name} (train)')
        axes[0].plot(epochs, hist['val_loss'],   '-',  color=color, linewidth=2.0,
                     label=f'{name} (val)')

        axes[1].plot(epochs, hist['val_acc'], '-', color=color, linewidth=2.0,
                     label=name)

    axes[0].set_title('Strata (Cross-Entropy Loss)')
    axes[0].set_xlabel('Epoka')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=9)

    axes[1].set_title('Dokładność walidacyjna (Val Accuracy)')
    axes[1].set_xlabel('Epoka')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_class_distribution(
    label_counts: pd.Series,
    title: str = 'Rozkład klas diagnostycznych PTB-XL',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Wykres słupkowy + kołowy rozkładu klas."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    colors = [CLASS_COLORS.get(c, '#607D8B') for c in label_counts.index]

    # Słupkowy
    bars = axes[0].bar(label_counts.index, label_counts.values,
                       color=colors, edgecolor='white', alpha=0.9)
    axes[0].set_xlabel('Klasa diagnostyczna')
    axes[0].set_ylabel('Liczba rekordów')
    axes[0].set_title('Liczba rekordów na klasę')
    for bar, val in zip(bars, label_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                     f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Kołowy
    axes[1].pie(label_counts.values, labels=label_counts.index,
                autopct='%1.1f%%', colors=colors, startangle=90,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title('Udział procentowy')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig


def plot_per_class_metrics(
    results: dict,
    class_names: List[str],
    metric: str = 'f1-score',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Heatmapa metryk per-klasa dla wszystkich modeli.

    Parametry
    ---------
    results : dict
        {nazwa_modelu: {'y_true': ndarray, 'y_pred': ndarray}}
    metric : str
        'f1-score', 'precision' lub 'recall'.
    """
    from sklearn.metrics import classification_report

    data = {}
    for name, res in results.items():
        report = classification_report(
            res['y_true'], res['y_pred'],
            target_names=class_names, output_dict=True, zero_division=0
        )
        data[name] = [report[cls][metric] for cls in class_names]

    df_heatmap = pd.DataFrame(data, index=class_names)

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 1.4), 4))
    sns.heatmap(
        df_heatmap, annot=True, fmt='.3f', cmap='YlOrRd',
        vmin=0, vmax=1, ax=ax,
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': metric}
    )
    ax.set_title(f'{metric.capitalize()} per klasa – porównanie modeli', fontweight='bold')
    ax.set_ylabel('Klasa diagnostyczna')
    ax.set_xlabel('Model')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig
