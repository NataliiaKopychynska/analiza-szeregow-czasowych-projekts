"""
Skrypt generujący notebooki Jupyter dla projektu klasyfikacji EKG.
Uruchomić jednokrotnie: python create_notebooks.py
"""
import json
import os

NOTEBOOKS_DIR = "notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pomocnicze funkcje do tworzenia komórek
# ─────────────────────────────────────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text,
    }

def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.9.0"},
        },
        "cells": cells,
    }

def save(name, cells):
    path = os.path.join(NOTEBOOKS_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f"Zapisano: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 – Eksploracyjna Analiza Danych (EDA)
# ═════════════════════════════════════════════════════════════════════════════

eda_cells = [

md("""# Notebook 1: Eksploracyjna Analiza Danych (EDA) – PTB-XL

## Cel
Zapoznanie się ze strukturą zbioru danych PTB-XL:
- analiza demograficzna pacjentów
- rozkład klas diagnostycznych (5 superklasowych kategorii)
- wizualizacja przykładowych sygnałów EKG (wszystkie 12 odprowadzeń)
- analiza długości sygnałów i brakujących danych

**Wymaganie:** Kontrola pośrednia nr 1
"""),

code("""\
# ── Instalacja zależności (jeśli potrzebna) ──────────────────────────────────
# Odkomentuj i uruchom raz:
# !pip install -r ../requirements.txt
"""),

code("""\
# ── Importy ──────────────────────────────────────────────────────────────────
import os
import sys
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import wfdb

# Dodaj katalog nadrzędny do ścieżki, żeby importować moduły z src/
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()), ''))
sys.path.insert(0, '..')
from src.data_loader import (
    load_ptbxl_metadata,
    load_scp_statements,
    build_labels,
    load_raw_data,
    SUPERCLASSES,
)

# Konfiguracja wizualizacji
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
sns.set_theme(style='whitegrid')

# ── Ścieżka do danych ─────────────────────────────────────────────────────────
# Zmień tę ścieżkę na lokalizację pobranego zbioru PTB-XL
DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'

print(f"Ścieżka danych: {DATA_PATH}")
print(f"Czy katalog istnieje: {os.path.isdir(DATA_PATH)}")
"""),

md("""## 1. Wczytanie metadanych

Plik `ptbxl_database.csv` zawiera informacje o każdym zapisie EKG:
- dane demograficzne pacjenta (wiek, płeć, wzrost, waga)
- kody diagnostyczne SCP w formacie słownika Python
- numer foldu walidacyjnego (`strat_fold` 1–10)
- ścieżki do plików sygnałów (100 Hz i 500 Hz)
"""),

code("""\
# Wczytaj metadane
Y = load_ptbxl_metadata(DATA_PATH)
print(f"Liczba rekordów EKG: {len(Y):,}")
print(f"\\nKolumny DataFrame:\\n{list(Y.columns)}")
print(f"\\nPierwsze 3 rekordy:")
Y.head(3)
"""),

code("""\
# Wczytaj i wyświetl kody SCP
agg_df = load_scp_statements(DATA_PATH)
print(f"Liczba kodów SCP: {len(agg_df)}")
print(f"\\nKolumny SCP statements:\\n{list(agg_df.columns)}")

# Pokaż superklasy diagnostyczne
diag_df = agg_df[agg_df.diagnostic == 1]
print(f"\\nKody diagnostyczne: {len(diag_df)}")
print("\\nRozkład po superklasach:")
print(diag_df['diagnostic_class'].value_counts())
"""),

md("""## 2. Przygotowanie etykiet

Agregujemy szczegółowe kody SCP do 5 superklasowych kategorii diagnostycznych:
- **NORM** – prawidłowy zapis EKG
- **MI**   – zawał mięśnia sercowego (Myocardial Infarction)
- **STTC** – zmiany odcinka ST i fali T (ST/T Change)
- **CD**   – zaburzenia przewodzenia (Conduction Disturbance)
- **HYP**  – przerost serca (Hypertrophy)
"""),

code("""\
# Buduj etykiety diagnostyczne
Y_labeled = build_labels(Y, DATA_PATH)
print(f"Rekordy z etykietami: {len(Y_labeled):,}")
print(f"\\nRozkład etykiet (single-label):")
label_counts = Y_labeled['label_single'].value_counts()
print(label_counts)
"""),

code("""\
# Wizualizacja rozkładu klas
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wykres słupkowy
colors = ['#2196F3', '#F44336', '#FF9800', '#4CAF50', '#9C27B0']
axes[0].bar(label_counts.index, label_counts.values, color=colors)
axes[0].set_title('Rozkład klas diagnostycznych (superklasy)', fontweight='bold')
axes[0].set_xlabel('Klasa diagnostyczna')
axes[0].set_ylabel('Liczba rekordów')
for i, (cls, cnt) in enumerate(label_counts.items()):
    axes[0].text(i, cnt + 50, str(cnt), ha='center', va='bottom', fontweight='bold')

# Wykres kołowy
axes[1].pie(
    label_counts.values, labels=label_counts.index,
    autopct='%1.1f%%', colors=colors, startangle=90
)
axes[1].set_title('Procentowy udział klas', fontweight='bold')

plt.tight_layout()
plt.savefig('../results/eda_class_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
print(f"\\nNiezbalansowanie klas: max/min = {label_counts.max()/label_counts.min():.2f}x")
"""),

md("""## 3. Analiza demograficzna pacjentów"""),

code("""\
# Podstawowe statystyki demograficzne
print("=== Statystyki demograficzne ===")
print(f"\\nWiek (lata):")
print(Y_labeled['age'].describe().round(1))

print(f"\\nPłeć (0=kobieta, 1=mężczyzna):")
sex_counts = Y_labeled['sex'].value_counts()
print(sex_counts)
print(f"  Kobiety: {sex_counts.get(0, 0)} ({100*sex_counts.get(0, 0)/len(Y_labeled):.1f}%)")
print(f"  Mężczyźni: {sex_counts.get(1, 0)} ({100*sex_counts.get(1, 0)/len(Y_labeled):.1f}%)")
"""),

code("""\
# Rozkład wieku i płci
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram wieku
axes[0].hist(Y_labeled['age'].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].axvline(Y_labeled['age'].median(), color='red', linestyle='--', label=f"Mediana: {Y_labeled['age'].median():.0f}")
axes[0].set_title('Rozkład wieku pacjentów', fontweight='bold')
axes[0].set_xlabel('Wiek [lata]')
axes[0].set_ylabel('Liczba rekordów')
axes[0].legend()

# Wiek wg klasy
Y_labeled.boxplot(column='age', by='label_single', ax=axes[1],
                  boxprops=dict(color='steelblue'),
                  medianprops=dict(color='red'))
axes[1].set_title('Rozkład wieku wg klasy diagnostycznej', fontweight='bold')
axes[1].set_xlabel('Klasa diagnostyczna')
axes[1].set_ylabel('Wiek [lata]')

# Płeć wg klasy
sex_class = Y_labeled.groupby(['label_single', 'sex']).size().unstack(fill_value=0)
sex_class.columns = ['Kobiety', 'Mężczyźni']
sex_class.plot(kind='bar', ax=axes[2], color=['#FF69B4', '#4169E1'], alpha=0.85)
axes[2].set_title('Płeć wg klasy diagnostycznej', fontweight='bold')
axes[2].set_xlabel('Klasa')
axes[2].set_ylabel('Liczba pacjentów')
axes[2].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('../results/eda_demographics.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 4. Wizualizacja sygnałów EKG

Wczytamy kilka przykładowych sygnałów EKG i zwizualizujemy wszystkie 12 odprowadzeń.

**12 odprowadzeń EKG:**
- Kończynowe: I, II, III, aVR, aVL, aVF
- Przedsercowe: V1, V2, V3, V4, V5, V6
"""),

code("""\
# Wczytaj kilka przykładowych sygnałów EKG (100 Hz)
sample_idx = Y_labeled.groupby('label_single').first().index
sample_df = Y_labeled.loc[sample_idx]
X_sample = load_raw_data(sample_df, sampling_rate=100, data_path=DATA_PATH)

print(f"Kształt danych EKG: {X_sample.shape}")
print(f"  Wymiar 0: liczba rekordów ({X_sample.shape[0]})")
print(f"  Wymiar 1: liczba próbek ({X_sample.shape[1]} = {X_sample.shape[1]/100:.0f}s przy 100 Hz)")
print(f"  Wymiar 2: liczba odprowadzeń ({X_sample.shape[2]})")
"""),

code("""\
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
SUPERCLASS_COLORS = {'NORM': '#2196F3', 'MI': '#F44336', 'STTC': '#FF9800', 'CD': '#4CAF50', 'HYP': '#9C27B0'}

def plot_ecg_12lead(signal, fs=100, title='', color='steelblue'):
    \"\"\"Rysuje pełny 12-odprowadzeniowy zapis EKG w układzie 4x3.\"\"\"
    fig, axes = plt.subplots(4, 3, figsize=(18, 10), sharex=True)
    t = np.arange(signal.shape[0]) / fs

    for idx, (ax, name) in enumerate(zip(axes.flatten(), LEAD_NAMES)):
        ax.plot(t, signal[:, idx], color=color, linewidth=0.8)
        ax.set_title(f'Odprowadzenie {name}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Amplituda [mV]', fontsize=8)
        ax.grid(True, alpha=0.3)
        if idx >= 9:
            ax.set_xlabel('Czas [s]', fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig

# Wyświetl po jednym przykładzie z każdej klasy
for i, (label, row) in enumerate(sample_df.iterrows()):
    cls = row['label_single']
    fig = plot_ecg_12lead(
        X_sample[i], fs=100,
        title=f'Klasa: {cls} | Pacjent: {int(row.patient_id)} | '
              f'Wiek: {int(row.age) if not pd.isna(row.age) else \"?\"} lat',
        color=SUPERCLASS_COLORS.get(cls, 'steelblue')
    )
    plt.savefig(f'../results/eda_ecg_{cls}.png', bbox_inches='tight', dpi=120)
    plt.show()
"""),

md("""## 5. Analiza długości sygnałów i brakujących danych"""),

code("""\
# Sprawdź brakujące wartości w metadanych
print("=== Analiza brakujących danych ===")
missing = Y_labeled.isnull().sum()
missing_pct = (missing / len(Y_labeled) * 100).round(2)
missing_df = pd.DataFrame({'Brakujące': missing, 'Procent': missing_pct})
print(missing_df[missing_df['Brakujące'] > 0].sort_values('Procent', ascending=False))
"""),

code("""\
# Analiza podziału na foldy (train/val/test)
print("=== Podział danych (strat_fold) ===")
fold_dist = Y_labeled['strat_fold'].value_counts().sort_index()
print(fold_dist)

train_size = len(Y_labeled[Y_labeled.strat_fold <= 8])
val_size   = len(Y_labeled[Y_labeled.strat_fold == 9])
test_size  = len(Y_labeled[Y_labeled.strat_fold == 10])
total = len(Y_labeled)

print(f"\\nTrening (fold 1-8): {train_size:5,} rekordów ({100*train_size/total:.1f}%)")
print(f"Walidacja (fold 9): {val_size:5,} rekordów ({100*val_size/total:.1f}%)")
print(f"Test (fold 10):     {test_size:5,} rekordów ({100*test_size/total:.1f}%)")
"""),

code("""\
# Rozkład klas w każdym zbiorze (sprawdź stratyfikację)
splits = {
    'Trening': Y_labeled[Y_labeled.strat_fold <= 8],
    'Walidacja': Y_labeled[Y_labeled.strat_fold == 9],
    'Test': Y_labeled[Y_labeled.strat_fold == 10],
}

print("=== Rozkład klas wg podziału ===")
for split_name, split_df in splits.items():
    counts = split_df['label_single'].value_counts()
    pcts = (counts / len(split_df) * 100).round(1)
    print(f"\\n{split_name}:")
    for cls in SUPERCLASSES:
        c = counts.get(cls, 0)
        p = pcts.get(cls, 0)
        print(f"  {cls:5s}: {c:5,} ({p:.1f}%)")
"""),

code("""\
print("\\n=== PODSUMOWANIE EDA ===")
print(f"Całkowita liczba rekordów EKG: {len(Y_labeled):,}")
print(f"Liczba unikalnych pacjentów:   {Y_labeled['patient_id'].nunique():,}")
print(f"Częstotliwość próbkowania:     100 Hz (lr) / 500 Hz (hr)")
print(f"Długość sygnału:               10 sekund (1000 próbek @ 100 Hz)")
print(f"Liczba odprowadzeń:            12")
print(f"Klasy diagnostyczne:           {', '.join(SUPERCLASSES)}")
print(f"Niezbalansowanie klas:         {label_counts.max()/label_counts.min():.1f}x")
"""),

]

save("01_eda.ipynb", eda_cells)


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 – Potok przetwarzania danych + ocena jakości
# ═════════════════════════════════════════════════════════════════════════════

preproc_cells = [

md("""# Notebook 2: Potok przetwarzania wstępnego (Preprocessing Pipeline)

## Cel
Implementacja kompletnego potoku przetwarzania sygnałów EKG:
1. Ocena jakości sygnału (SNR, clipping, segmenty płaskie)
2. Usunięcie dryftu bazowego (baseline wander)
3. Filtracja pasmowoprzepustowa [0.5–40 Hz]
4. Filtr notch 50 Hz (zakłócenia sieciowe)
5. Normalizacja (Z-score, Min-Max, Robust)
6. Analiza wpływu pochodnych sygnału
7. Ekstrakcja cech statystycznych dla klasycznych modeli ML
8. Zapis przetworzonych danych

**Wymaganie:** Kontrola pośrednia nr 2 – kompletny potok + ocena jakości
"""),

code("""\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq

sys.path.insert(0, '..')
from src.data_loader import (
    load_ptbxl_metadata, build_labels, load_raw_data,
    get_train_val_test_split, SUPERCLASSES,
)
from src.preprocessing import (
    assess_signal_quality,
    bandpass_filter, notch_filter, remove_baseline_wander,
    resample_signal, normalize_signal,
    compute_derivatives, preprocess_pipeline, preprocess_batch,
)
from src.feature_extraction import (
    extract_statistical_features, extract_features_batch,
    extract_features_with_derivatives, get_feature_names,
)

plt.rcParams['figure.dpi'] = 100
sns_colors = plt.cm.tab10.colors

DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
FS = 100   # częstotliwość próbkowania (niższe: 100 Hz)
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

print("Moduły załadowane pomyślnie.")
"""),

code("""\
# Wczytaj metadane i etykiety
Y = load_ptbxl_metadata(DATA_PATH)
Y = build_labels(Y, DATA_PATH)

train_idx, val_idx, test_idx = get_train_val_test_split(Y)
print(f"Trening: {len(train_idx):,} | Walidacja: {len(val_idx):,} | Test: {len(test_idx):,}")

# Wczytaj mały podzbiór do demonstracji (50 rekordów z każdej klasy)
demo_idx = []
for cls in SUPERCLASSES:
    cls_idx = Y[(Y.label_single == cls) & Y.index.isin(train_idx)].index[:50]
    demo_idx.extend(cls_idx)

Y_demo = Y.loc[demo_idx]
X_demo_raw = load_raw_data(Y_demo, sampling_rate=FS, data_path=DATA_PATH)
y_demo = Y_demo['label_single'].values
print(f"\\nDemo: {X_demo_raw.shape} | Klasy: {np.unique(y_demo)}")
"""),

md("""## 1. Ocena jakości sygnałów

Przed przetwarzaniem sprawdzamy jakość każdego sygnału. Automatyczna ocena
pozwala wykryć i odfiltrować sygnały niskiej jakości (artefakty, brak kontaktu
elektrody, nasycenie przetwornika).
"""),

code("""\
# Oblicz metryki jakości dla podzbioru demo
quality_results = []
for i, ecg in enumerate(X_demo_raw):
    q = assess_signal_quality(ecg, fs=FS)
    q['class'] = y_demo[i]
    quality_results.append(q)

df_quality = pd.DataFrame(quality_results)
print("=== Statystyki jakości sygnałów ===")
print(df_quality[['snr_db', 'clipping_ratio', 'flat_ratio', 'baseline_drift', 'quality_score']].describe().round(3))
"""),

code("""\
# Histogramy metryk jakości
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

metrics = [
    ('snr_db', 'SNR [dB]', 'Stosunek sygnału do szumu'),
    ('clipping_ratio', 'Clipping Ratio', 'Udział nasyconych próbek'),
    ('flat_ratio', 'Flat Ratio', 'Udział płaskich segmentów'),
    ('quality_score', 'Quality Score', 'Łączna ocena jakości [0-1]'),
]

for ax, (col, xlabel, title) in zip(axes.flatten(), metrics):
    ax.hist(df_quality[col], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(df_quality[col].median(), color='red', linestyle='--',
               label=f'Mediana: {df_quality[col].median():.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Liczba sygnałów')
    ax.set_title(title, fontweight='bold')
    ax.legend()

plt.tight_layout()
plt.savefig('../results/preprocessing_quality_histograms.png', bbox_inches='tight', dpi=150)
plt.show()

low_quality = (df_quality['quality_score'] < 0.5).sum()
print(f"\\nSygnały niskiej jakości (score < 0.5): {low_quality} ({100*low_quality/len(df_quality):.1f}%)")
"""),

md("""## 2. Wizualizacja efektów filtracji

Porównanie sygnału przed i po każdym kroku przetwarzania wstępnego.
"""),

code("""\
# Wybierz przykładowy sygnał (odprowadzenie II – najczęściej używane)
example_ecg = X_demo_raw[0].copy()
t = np.arange(len(example_ecg)) / FS

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

# 1. Surowy sygnał
axes[0].plot(t, example_ecg[:, 1], 'b', linewidth=0.8)
axes[0].set_title('1. Sygnał surowy (odprowadzenie II)', fontweight='bold')
axes[0].set_ylabel('Amplituda [mV]')

# 2. Po usunięciu dryftu bazowego
ecg_no_baseline = remove_baseline_wander(example_ecg, fs=FS)
axes[1].plot(t, ecg_no_baseline[:, 1], 'g', linewidth=0.8)
axes[1].set_title('2. Po usunięciu dryftu bazowego (filtr medianowy 200ms + 600ms)', fontweight='bold')
axes[1].set_ylabel('Amplituda [mV]')

# 3. Po filtrze pasmowoprzepustowym
ecg_bp = bandpass_filter(ecg_no_baseline, fs=FS)
axes[2].plot(t, ecg_bp[:, 1], 'darkorange', linewidth=0.8)
axes[2].set_title('3. Po filtrze pasmowoprzepustowym [0.5–40 Hz] Butterworth 4. rzędu', fontweight='bold')
axes[2].set_ylabel('Amplituda [mV]')

# 4. Po filtrze notch 50 Hz
ecg_notch = notch_filter(ecg_bp, fs=FS)
axes[3].plot(t, ecg_notch[:, 1], 'purple', linewidth=0.8)
axes[3].set_title('4. Po filtrze notch 50 Hz (usunięcie zakłóceń sieciowych)', fontweight='bold')
axes[3].set_ylabel('Amplituda [mV]')

# 5. Po normalizacji Z-score
ecg_norm = normalize_signal(ecg_notch, method='zscore')
axes[4].plot(t, ecg_norm[:, 1], 'red', linewidth=0.8)
axes[4].set_title('5. Po normalizacji Z-score (μ=0, σ=1)', fontweight='bold')
axes[4].set_ylabel('Amplituda [j.u.]')
axes[4].set_xlabel('Czas [s]')

plt.tight_layout()
plt.savefig('../results/preprocessing_steps.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

code("""\
# Porównanie widm częstotliwościowych przed i po filtracji
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (sig, label, color) in zip(axes, [
    (example_ecg[:, 1], 'Surowy sygnał', 'blue'),
    (ecg_notch[:, 1], 'Po filtracji', 'red'),
]):
    freqs = rfftfreq(len(sig), d=1.0/FS)
    power = np.abs(rfft(sig)) ** 2
    ax.semilogy(freqs, power, color=color, alpha=0.8, linewidth=1.0)
    ax.axvline(0.5,  color='gray', linestyle=':', alpha=0.7, label='0.5 Hz (dolne odcięcie)')
    ax.axvline(40.0, color='gray', linestyle='--', alpha=0.7, label='40 Hz (górne odcięcie)')
    ax.axvline(50.0, color='orange', linestyle='-.',  alpha=0.9, label='50 Hz (zakłócenia sieciowe)')
    ax.set_title(f'Widmo mocy: {label}', fontweight='bold')
    ax.set_xlabel('Częstotliwość [Hz]')
    ax.set_ylabel('Moc [log]')
    ax.set_xlim([0, FS/2])
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('../results/preprocessing_spectra.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 3. Porównanie metod normalizacji"""),

code("""\
# Porównaj trzy metody normalizacji
methods = ['zscore', 'minmax', 'robust']
colors = ['red', 'green', 'purple']
labels_norm = ['Z-score (μ=0, σ=1)', 'Min-Max ([0,1])', 'Robust (mediana/IQR)']

fig, axes = plt.subplots(len(methods), 1, figsize=(14, 10), sharex=True)
for ax, method, color, lbl in zip(axes, methods, colors, labels_norm):
    normed = normalize_signal(ecg_bp, method=method)
    ax.plot(t, normed[:, 1], color=color, linewidth=0.8)
    ax.set_title(f'Normalizacja: {lbl}', fontweight='bold')
    ax.set_ylabel('Amplituda')
    stats_str = f'min={normed[:,1].min():.2f}, max={normed[:,1].max():.2f}, std={normed[:,1].std():.2f}'
    ax.text(0.98, 0.05, stats_str, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='gray')
axes[-1].set_xlabel('Czas [s]')

plt.tight_layout()
plt.savefig('../results/preprocessing_normalization_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 4. Analiza pochodnych sygnału

Pochodne sygnału EKG podkreślają szybkie zmiany amplitudy, co jest pomocne
w detekcji kompleksów QRS i może poprawiać wyniki klasyfikacji.
"""),

code("""\
# Oblicz i zwizualizuj pochodne sygnału
ecg_processed = preprocess_pipeline(example_ecg, fs=FS)
first_deriv, second_deriv = compute_derivatives(ecg_processed)

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
for ax, sig, title, color in zip(axes,
    [ecg_processed[:, 1], first_deriv[:, 1], second_deriv[:, 1]],
    ['Sygnał przetworzone (odprowadzenie II)',
     '1. pochodna sygnału (podkreśla kompleksy QRS)',
     '2. pochodna sygnału (punkty przegięcia)'],
    ['steelblue', 'darkorange', 'green']
):
    ax.plot(t, sig, color=color, linewidth=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Amplituda')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('Czas [s]')

plt.tight_layout()
plt.savefig('../results/preprocessing_derivatives.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 5. Przetwarzanie pełnego zbioru danych"""),

code("""\
# Przetwórz cały zbiór demo przez potok
print("Przetwarzanie potoku dla zbioru demo...")
X_demo_processed = preprocess_batch(X_demo_raw, fs=FS, verbose=True)

print(f"\\nKształt przetworzonego zbioru: {X_demo_processed.shape}")
print(f"Zakres wartości: [{X_demo_processed.min():.3f}, {X_demo_processed.max():.3f}]")
print(f"Średnia: {X_demo_processed.mean():.6f}")
print(f"Odch. std: {X_demo_processed.std():.4f}")
"""),

md("""## 6. Ekstrakcja cech dla klasycznych modeli ML

Dla klasycznych algorytmów ML (SVM, RF, LR, KNN, GB) nie możemy bezpośrednio
używać surowych sygnałów EKG (1000×12 = 12 000 wartości). Zamiast tego
ekstrahujemy wektor **228 cech** statystycznych (19 cech × 12 odprowadzeń).
"""),

code("""\
# Ekstrakcja cech ze zbioru demo
print("Ekstrakcja cech statystycznych...")
X_demo_features = extract_features_batch(X_demo_processed, fs=FS, verbose=True)

print(f"\\nKształt macierzy cech: {X_demo_features.shape}")
print(f"  {X_demo_features.shape[0]} próbek × {X_demo_features.shape[1]} cech")

# Nazwy cech
feature_names = get_feature_names()
print(f"\\nPierwsze 5 nazw cech: {feature_names[:5]}")
print(f"Ostatnie 5 nazw cech: {feature_names[-5:]}")
"""),

code("""\
# Ekstrakcja cech z pochodną (456 cech = 228×2)
print("Ekstrakcja cech z pochodną...")
X_demo_features_deriv = extract_features_batch(
    X_demo_processed, fs=FS, use_derivatives=True, verbose=True
)
print(f"\\nKształt z pochodnymi: {X_demo_features_deriv.shape}")
print(f"  (228 cech oryginalnych + 228 cech pochodnej = {X_demo_features_deriv.shape[1]})")
"""),

code("""\
# Sprawdź brakujące wartości i wartości nieskończone w cechach
print("=== Kontrola jakości cech ===")
nan_count = np.isnan(X_demo_features).sum()
inf_count = np.isinf(X_demo_features).sum()
print(f"NaN w cechach:        {nan_count}")
print(f"Inf w cechach:        {inf_count}")

if nan_count > 0 or inf_count > 0:
    X_demo_features = np.nan_to_num(X_demo_features, nan=0.0, posinf=0.0, neginf=0.0)
    print("Naprawiono: zastąpiono NaN/Inf zerami")

print(f"\\nStatystyki cech:")
print(f"  Min: {X_demo_features.min():.4f}")
print(f"  Max: {X_demo_features.max():.4f}")
print(f"  Średnia: {X_demo_features.mean():.4f}")
print(f"  Odch. std: {X_demo_features.std():.4f}")
"""),

code("""\
# Zapis przetworzonych danych do katalogu data/processed/
os.makedirs('../data/processed', exist_ok=True)

# Zapis zbioru demo (pełny zbiór wymaga więcej czasu – patrz Notebook 3)
np.save('../data/processed/X_demo_processed.npy', X_demo_processed)
np.save('../data/processed/X_demo_features.npy', X_demo_features)
np.save('../data/processed/X_demo_features_deriv.npy', X_demo_features_deriv)
np.save('../data/processed/y_demo.npy', y_demo)

print("Zapisano pliki:")
print("  data/processed/X_demo_processed.npy    – przetworzone sygnały (demo)")
print("  data/processed/X_demo_features.npy     – cechy 228-dim (demo)")
print("  data/processed/X_demo_features_deriv.npy – cechy 456-dim z pochodnymi (demo)")
print("  data/processed/y_demo.npy              – etykiety (demo)")
"""),

code("""\
print("\\n=== PODSUMOWANIE POTOKU PRZETWARZANIA ===")
print("Kroki potoku:")
print("  1. Resampling (jeśli potrzebny)")
print("  2. Usunięcie dryftu bazowego (filtr medianowy)")
print("  3. Filtr pasmowoprzepustowy [0.5-40 Hz] Butterworth 4. rzędu")
print("  4. Filtr notch 50 Hz")
print("  5. Normalizacja Z-score (kanał po kanale)")
print()
print("Ocena jakości: SNR, clipping, flat ratio, baseline drift")
print(f"Ekstrakcja cech: 228 cech (19 × 12 odprowadzeń)")
print(f"Z pochodnymi:    456 cech (228 + 228)")
"""),

]

save("02_preprocessing.ipynb", preproc_cells)


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 – Klasyczne metody uczenia maszynowego
# ═════════════════════════════════════════════════════════════════════════════

ml_cells = [

md("""# Notebook 3: Klasyczne metody uczenia maszynowego (ML)

## Cel
Implementacja i ocena 5 klasycznych algorytmów ML do klasyfikacji EKG:
1. **Regresja logistyczna** (Logistic Regression)
2. **Las losowy** (Random Forest)
3. **Maszyna wektorów nośnych** (SVM z jądrem RBF)
4. **k-Nearest Neighbors** (KNN, k=5)
5. **Gradient Boosting**
6. **Naiwny Bayes** (model bazowy – punkt odniesienia)

Dane wejściowe: wektory 228 cech statystycznych (z Notebook 2).
Ocena bez strojenia hiperparametrów (Kontrola 2).

**Wymaganie:** Kontrola pośrednia nr 2 – wszystkie algorytmy (bez optymalizacji)
"""),

code("""\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, ConfusionMatrixDisplay,
)

sys.path.insert(0, '..')
from src.data_loader import (
    load_ptbxl_metadata, build_labels, load_raw_data,
    get_train_val_test_split, SUPERCLASSES,
)
from src.preprocessing import preprocess_pipeline, preprocess_batch
from src.feature_extraction import extract_features_batch, get_feature_names
from src.classical_models import get_classical_models, evaluate_all_models, train_evaluate_model

plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid')

DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
FS = 100

print("Moduły załadowane.")
"""),

md("""## 1. Wczytanie i przygotowanie danych

Wczytujemy dane, przetwarzamy je potok przetwarzania i ekstrahujemy cechy.
Używamy podzbioru 500 próbek (100 z każdej klasy) do szybkiej demonstracji.
Dla pełnego eksperymentu można usunąć limit `n_per_class`.
"""),

code("""\
# Wczytaj metadane
Y = load_ptbxl_metadata(DATA_PATH)
Y = build_labels(Y, DATA_PATH)
train_idx, val_idx, test_idx = get_train_val_test_split(Y)

# ── Podzbiór do demonstracji (100 próbek na klasę) ────────────────────────
# Zmień n_per_class=None, żeby użyć PEŁNEGO zbioru (zajmie kilka minut)
N_PER_CLASS = 100

def sample_per_class(df, idx, n=None):
    sampled = []
    for cls in SUPERCLASSES:
        cls_idx = df[(df.label_single == cls) & df.index.isin(idx)].index
        sampled.extend(cls_idx[:n] if n else cls_idx)
    return df.loc[sampled]

Y_train_s = sample_per_class(Y, train_idx, N_PER_CLASS)
Y_test_s  = sample_per_class(Y, test_idx,  N_PER_CLASS // 5)

print(f"Podzbiór treningowy: {len(Y_train_s):,} próbek")
print(f"Podzbiór testowy:    {len(Y_test_s):,} próbek")
"""),

code("""\
# Wczytaj surowe sygnały i przetwórz
print("Ładowanie sygnałów...")
X_train_raw = load_raw_data(Y_train_s, sampling_rate=FS, data_path=DATA_PATH)
X_test_raw  = load_raw_data(Y_test_s,  sampling_rate=FS, data_path=DATA_PATH)

print("Przetwarzanie potoku...")
X_train_proc = preprocess_batch(X_train_raw, fs=FS, verbose=False)
X_test_proc  = preprocess_batch(X_test_raw,  fs=FS, verbose=False)

print("Ekstrakcja cech...")
X_train = extract_features_batch(X_train_proc, fs=FS, verbose=False)
X_test  = extract_features_batch(X_test_proc,  fs=FS, verbose=False)

# Napraw NaN/Inf
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

# Koduj etykiety jako liczby całkowite
le = LabelEncoder()
le.fit(SUPERCLASSES)
y_train = le.transform(Y_train_s['label_single'].values)
y_test  = le.transform(Y_test_s['label_single'].values)

print(f"\\nX_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
print(f"Klasy: {le.classes_}")
"""),

md("""## 2. Trening i ewaluacja modeli"""),

code("""\
# Uruchom wszystkie modele i zbierz wyniki
summary_df, all_results = evaluate_all_models(
    X_train, y_train, X_test, y_test,
    classes=list(le.classes_)
)
"""),

md("""## 3. Porównanie wyników"""),

code("""\
# Tabela porównawcza
print("\\n=== TABELA PORÓWNAWCZA MODELI ===")
display_cols = ['Model', 'Accuracy', 'Balanced Accuracy', 'F1 Macro', 'F1 Weighted', 'Train Time [s]']
print(summary_df[display_cols].to_string(index=False))
"""),

code("""\
# Wykres słupkowy metryk
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
models = summary_df['Model'].tolist()
x = np.arange(len(models))
width = 0.35

# F1-score
axes[0].bar(x - width/2, summary_df['F1 Macro'],    width, label='F1 Macro',    color='steelblue', alpha=0.85)
axes[0].bar(x + width/2, summary_df['F1 Weighted'], width, label='F1 Weighted', color='coral',     alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=25, ha='right')
axes[0].set_ylabel('F1-score')
axes[0].set_title('Porównanie F1-score modeli', fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 1)

# Accuracy vs. Balanced Accuracy
axes[1].bar(x - width/2, summary_df['Accuracy'],          width, label='Accuracy',          color='forestgreen', alpha=0.85)
axes[1].bar(x + width/2, summary_df['Balanced Accuracy'], width, label='Balanced Accuracy', color='orange',      alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=25, ha='right')
axes[1].set_ylabel('Dokładność')
axes[1].set_title('Accuracy vs Balanced Accuracy', fontweight='bold')
axes[1].legend()
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('../results/classical_ml_comparison.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 4. Macierze pomyłek (Confusion Matrices)"""),

code("""\
# Macierze pomyłek dla wszystkich modeli
n_models = len(all_results)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for ax, (name, result) in zip(axes.flatten(), all_results.items()):
    cm = result['confusion_matrix']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title(f'{name}\\nAcc={result["accuracy"]:.3f} | F1={result["f1_macro"]:.3f}',
                 fontweight='bold', fontsize=10)

# Ukryj ostatni pusty wykres jeśli modeli jest mniej niż 6
for i in range(len(all_results), 6):
    axes.flatten()[i].set_visible(False)

plt.tight_layout()
plt.savefig('../results/classical_ml_confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 5. Analiza ważności cech (Random Forest)

Las losowy dostarcza miarę ważności cech (feature importance) – które cechy
statystyczne są najbardziej pomocne w klasyfikacji.
"""),

code("""\
# Feature importance z Random Forest
rf_model = all_results['Random Forest']['model']
importances = rf_model.feature_importances_
feature_names = get_feature_names()

# Top 20 najważniejszych cech
top_n = 20
top_idx = np.argsort(importances)[::-1][:top_n]

fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(
    [feature_names[i] for i in top_idx[::-1]],
    importances[top_idx[::-1]],
    color='steelblue', alpha=0.85
)
ax.set_xlabel('Ważność cechy (Gini impurity)')
ax.set_title(f'Top {top_n} najważniejszych cech – Random Forest', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../results/classical_ml_feature_importance.png', bbox_inches='tight', dpi=150)
plt.show()

print("\\nTop 10 najważniejszych cech:")
for rank, i in enumerate(top_idx[:10], 1):
    print(f"  {rank:2d}. {feature_names[i]:35s} = {importances[i]:.4f}")
"""),

code("""\
# Zapis najlepszego modelu
import pickle
best_model_name = summary_df.iloc[0]['Model']
best_model = all_results[best_model_name]['model']

os.makedirs('../models', exist_ok=True)
with open('../models/best_classical_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'model_name': best_model_name, 'label_encoder': le}, f)

print(f"Najlepszy model: {best_model_name}")
print(f"  F1 Macro: {summary_df.iloc[0]['F1 Macro']:.4f}")
print(f"  Accuracy: {summary_df.iloc[0]['Accuracy']:.4f}")
print(f"Zapisano do: models/best_classical_model.pkl")
"""),

]

save("03_classical_ml.ipynb", ml_cells)


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 4 – Głębokie sieci neuronowe
# ═════════════════════════════════════════════════════════════════════════════

dl_cells = [

md("""# Notebook 4: Głębokie sieci neuronowe (Deep Learning)

## Cel
Implementacja i porównanie 3 architektur głębokich sieci neuronowych:
1. **CNN1D** – prosta sieć konwolucyjna 1D (3 bloki konwolucyjne + 2 FC)
2. **ResNet1D** – głęboka sieć rezydualna (He et al., 2016, adaptacja 1D)
3. **BiLSTM** – dwukierunkowa sieć LSTM z mechanizmem uwagi (attention)

Dane wejściowe: surowe przetworzone sygnały EKG (1000×12) → (12, 1000) dla Conv1d.
Ocena bez strojenia hiperparametrów (Kontrola 2).

**Wymaganie:** Kontrola pośrednia nr 2 – wszystkie architektury DL (bez optymalizacji)
"""),

code("""\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, ConfusionMatrixDisplay,
)

sys.path.insert(0, '..')
from src.data_loader import (
    load_ptbxl_metadata, build_labels, load_raw_data,
    get_train_val_test_split, SUPERCLASSES,
)
from src.preprocessing import preprocess_batch
from src.deep_models import (
    ECGDataset, CNN1D, ResNet1D, BiLSTMClassifier,
    Trainer, get_device, get_deep_models,
)

plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid')

DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
FS = 100
DEVICE = get_device()

print(f"PyTorch {torch.__version__}")
print(f"Urządzenie obliczeniowe: {DEVICE}")
"""),

md("""## 1. Przygotowanie danych

Wczytujemy przetworzone sygnały EKG i tworzymy PyTorch DataLoader'y.
"""),

code("""\
# Wczytaj metadane i etykiety
Y = load_ptbxl_metadata(DATA_PATH)
Y = build_labels(Y, DATA_PATH)
train_idx, val_idx, test_idx = get_train_val_test_split(Y)

# ── Podzbiór do demonstracji ────────────────────────────────────────────────
# Zmień N_PER_CLASS=None dla pełnego zbioru (trening ~1-2h na CPU)
N_PER_CLASS = 80   # 80 przykładów na klasę w treningu

def sample_per_class(df, idx, n=None):
    sampled = []
    for cls in SUPERCLASSES:
        cls_idx = df[(df.label_single == cls) & df.index.isin(idx)].index
        sampled.extend(cls_idx[:n] if n else cls_idx)
    return df.loc[sampled]

Y_train_s = sample_per_class(Y, train_idx, N_PER_CLASS)
Y_val_s   = sample_per_class(Y, val_idx,   N_PER_CLASS // 4)
Y_test_s  = sample_per_class(Y, test_idx,  N_PER_CLASS // 4)

print(f"Trening:   {len(Y_train_s):,} | Walidacja: {len(Y_val_s):,} | Test: {len(Y_test_s):,}")
"""),

code("""\
# Wczytaj surowe sygnały
print("Ładowanie i przetwarzanie sygnałów...")
def load_and_process(Y_df):
    X_raw = load_raw_data(Y_df, sampling_rate=FS, data_path=DATA_PATH)
    X_proc = preprocess_batch(X_raw, fs=FS, verbose=False)
    return X_proc

X_train = load_and_process(Y_train_s)
X_val   = load_and_process(Y_val_s)
X_test  = load_and_process(Y_test_s)

# Koduj etykiety
le = LabelEncoder()
le.fit(SUPERCLASSES)
y_train = le.transform(Y_train_s['label_single'].values)
y_val   = le.transform(Y_val_s['label_single'].values)
y_test  = le.transform(Y_test_s['label_single'].values)

print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
print(f"Klasy: {le.classes_}")
"""),

code("""\
# Utwórz DataLoader'y PyTorch
BATCH_SIZE = 32

train_dataset = ECGDataset(X_train, y_train)
val_dataset   = ECGDataset(X_val,   y_val)
test_dataset  = ECGDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Sprawdź kształt batcha
X_batch, y_batch = next(iter(train_loader))
print(f"Kształt batcha wejściowego: {X_batch.shape}")
print(f"  (batch={X_batch.shape[0]}, leads={X_batch.shape[1]}, samples={X_batch.shape[2]})")
print(f"Kształt etykiet: {y_batch.shape}")
"""),

md("""## 2. Architektura 1: CNN1D

Prosta sieć konwolucyjna 1D złożona z 3 bloków konwolucyjnych,
z BatchNorm, ReLU i MaxPool po każdym bloku.
"""),

code("""\
# Inicjalizacja modelu CNN1D
cnn_model = CNN1D(num_channels=12, num_classes=len(SUPERCLASSES))

# Wyświetl architekturę
total_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
print(f"CNN1D – liczba parametrów: {total_params:,}")
print()
print(cnn_model)
"""),

code("""\
# Trening CNN1D
print("\\n=== Trening CNN1D ===")
EPOCHS = 20
LR = 1e-3

cnn_trainer = Trainer(cnn_model, device=DEVICE, learning_rate=LR)
cnn_history = cnn_trainer.fit(train_loader, val_loader, epochs=EPOCHS, verbose=True)
"""),

md("""## 3. Architektura 2: ResNet1D

Głęboka sieć rezydualna adaptowana dla sygnałów 1D. Połączenia skip-connection
(Skip connections / Residual connections) umożliwiają trening bardzo głębokich
sieci bez problemu zanikającego gradientu.
"""),

code("""\
# Inicjalizacja modelu ResNet1D
resnet_model = ResNet1D(num_channels=12, num_classes=len(SUPERCLASSES))

total_params = sum(p.numel() for p in resnet_model.parameters() if p.requires_grad)
print(f"ResNet1D – liczba parametrów: {total_params:,}")
print()
print(resnet_model)
"""),

code("""\
# Trening ResNet1D
print("\\n=== Trening ResNet1D ===")
resnet_trainer = Trainer(resnet_model, device=DEVICE, learning_rate=LR)
resnet_history = resnet_trainer.fit(train_loader, val_loader, epochs=EPOCHS, verbose=True)
"""),

md("""## 4. Architektura 3: Bidirectional LSTM

Dwukierunkowa sieć LSTM z mechanizmem uwagi (attention). LSTM modeluje
długoterminowe zależności w sygnale; mechanizm uwagi pozwala sieci skupić się
na najważniejszych momentach czasowych (np. kompleksach QRS).
"""),

code("""\
# Inicjalizacja modelu BiLSTM
lstm_model = BiLSTMClassifier(
    input_size=12, hidden_size=128,
    num_layers=2, num_classes=len(SUPERCLASSES)
)

total_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"BiLSTM – liczba parametrów: {total_params:,}")
print()
print(lstm_model)
"""),

code("""\
# Trening BiLSTM
print("\\n=== Trening BiLSTM ===")
lstm_trainer = Trainer(lstm_model, device=DEVICE, learning_rate=LR)
lstm_history = lstm_trainer.fit(train_loader, val_loader, epochs=EPOCHS, verbose=True)
"""),

md("""## 5. Krzywe uczenia (Loss i Accuracy)"""),

code("""\
# Wizualizacja krzywych uczenia dla wszystkich 3 modeli
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models_hist = {
    'CNN1D':    (cnn_history,    'steelblue'),
    'ResNet1D': (resnet_history, 'darkorange'),
    'BiLSTM':   (lstm_history,   'green'),
}

# Strata (Loss)
for name, (hist, color) in models_hist.items():
    axes[0].plot(hist['train_loss'], '--', color=color, alpha=0.7, label=f'{name} train')
    axes[0].plot(hist['val_loss'],   '-',  color=color, linewidth=2, label=f'{name} val')

axes[0].set_title('Krzywe straty (Loss)', fontweight='bold')
axes[0].set_xlabel('Epoka')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend(fontsize=8)

# Dokładność (Accuracy)
for name, (hist, color) in models_hist.items():
    axes[1].plot(hist['val_acc'], '-', color=color, linewidth=2, label=name)

axes[1].set_title('Dokładność walidacyjna (Val Accuracy)', fontweight='bold')
axes[1].set_xlabel('Epoka')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('../results/deep_learning_training_curves.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

md("""## 6. Ewaluacja na zbiorze testowym"""),

code("""\
# Ewaluacja wszystkich modeli na zbiorze testowym
results_dl = {}

for name, trainer in [
    ('CNN1D', cnn_trainer),
    ('ResNet1D', resnet_trainer),
    ('BiLSTM', lstm_trainer)
]:
    y_true, y_pred = trainer.predict(test_loader)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm  = confusion_matrix(y_true, y_pred)

    results_dl[name] = {'y_true': y_true, 'y_pred': y_pred,
                        'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
                        'confusion_matrix': cm}

    print(f"\\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Macro:    {f1m:.4f}")
    print(f"F1 Weighted: {f1w:.4f}")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))
"""),

code("""\
# Macierze pomyłek DL
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, result) in zip(axes, results_dl.items()):
    disp = ConfusionMatrixDisplay(result['confusion_matrix'], display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap='Blues', values_format='d')
    ax.set_title(f'{name}\\nAcc={result["accuracy"]:.3f} | F1={result["f1_macro"]:.3f}',
                 fontweight='bold')

plt.tight_layout()
plt.savefig('../results/deep_learning_confusion_matrices.png', bbox_inches='tight', dpi=150)
plt.show()
"""),

code("""\
# Zbiorcze porównanie modeli DL
print("\\n=== PORÓWNANIE ARCHITEKTUR DEEP LEARNING ===")
summary_dl = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': r['accuracy'],
        'F1 Macro': r['f1_macro'],
        'F1 Weighted': r['f1_weighted'],
        'Parametrów': sum(p.numel() for p in m.parameters() if p.requires_grad),
    }
    for (name, r), m in zip(
        results_dl.items(),
        [cnn_model, resnet_model, lstm_model]
    )
])
print(summary_dl.to_string(index=False))
"""),

code("""\
# Zapis modeli
os.makedirs('../models', exist_ok=True)

torch.save(cnn_model.state_dict(),    '../models/cnn1d.pt')
torch.save(resnet_model.state_dict(), '../models/resnet1d.pt')
torch.save(lstm_model.state_dict(),   '../models/bilstm.pt')

print("Zapisano modele:")
print("  models/cnn1d.pt")
print("  models/resnet1d.pt")
print("  models/bilstm.pt")
"""),

code("""\
print("\\n=== PODSUMOWANIE – DEEP LEARNING ===")
print(f"\\nZaimplementowane architektury:")
print(f"  1. CNN1D    – prosta sieć konwolucyjna 1D")
print(f"  2. ResNet1D – sieć rezydualna (skip connections)")
print(f"  3. BiLSTM   – dwukierunkowy LSTM z mechanizmem uwagi")
print(f"\\nTrening: {EPOCHS} epok, LR={LR}, batch={BATCH_SIZE}")
print(f"Optymalizator: Adam + ReduceLROnPlateau")
print(f"Regularyzacja: Dropout, BatchNorm, Weight Decay, Gradient Clipping")
"""),

]

save("04_deep_learning.ipynb", dl_cells)


# ═════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 5 – Wizualizacje (standalone, do prezentacji)
# ═════════════════════════════════════════════════════════════════════════════

viz_cells = [

md("""# Notebook 5: Wizualizacje – Dashboard prezentacyjny

Notebook generuje wszystkie wykresy potrzebne do prezentacji projektu.
Można go uruchomić **po** Notebooku 02, 03 i 04 – wczytuje zapisane wyniki.

Zawiera:
- Porównanie sygnałów EKG wg klasy diagnostycznej
- Cały potok przetwarzania krok po kroku
- Widma częstotliwościowe przed/po filtracji
- Porównanie metod normalizacji
- Pochodne sygnału
- Dashboard oceny jakości
- Porównanie wszystkich modeli ML i DL
- Macierze pomyłek
- Ważność cech (Feature Importance)
- Heatmapa F1-score per klasa
"""),

code("""\
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120

sys.path.insert(0, '..')
from src.data_loader import (
    load_ptbxl_metadata, build_labels, load_raw_data,
    get_train_val_test_split, SUPERCLASSES,
)
from src.preprocessing import (
    preprocess_pipeline, preprocess_batch,
    bandpass_filter, notch_filter, remove_baseline_wander,
)
from src.visualization import (
    plot_ecg_12lead, plot_ecg_comparison,
    plot_preprocessing_steps, plot_spectra_comparison,
    plot_normalization_comparison, plot_derivatives,
    plot_quality_dashboard, plot_confusion_matrices,
    plot_model_comparison, plot_feature_importance,
    plot_training_curves, plot_class_distribution,
    plot_per_class_metrics, CLASS_COLORS,
)
from src.preprocessing import assess_signal_quality

os.makedirs('../results', exist_ok=True)

DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
FS = 100
print("Gotowe. Ustaw DATA_PATH jeśli inne niż domyślne.")
"""),

md("""## 1. Wczytanie danych (po jednym przykładzie z każdej klasy)"""),

code("""\
Y = load_ptbxl_metadata(DATA_PATH)
Y = build_labels(Y, DATA_PATH)
train_idx, _, _ = get_train_val_test_split(Y)

# Po jednym przykładzie z każdej klasy
demo_rows = {}
for cls in SUPERCLASSES:
    idx = Y[(Y.label_single == cls) & Y.index.isin(train_idx)].index[0]
    demo_rows[cls] = Y.loc[idx]

demo_df = Y.loc[list(demo_rows[cls].name if hasattr(demo_rows[cls], 'name') else k
                     for k, _ in demo_rows.items())]
# prostsze podejście:
demo_df = pd.concat([
    Y[(Y.label_single == cls) & Y.index.isin(train_idx)].head(1)
    for cls in SUPERCLASSES
])

X_raw_demo = load_raw_data(demo_df, sampling_rate=FS, data_path=DATA_PATH)
print(f"Wczytano {len(X_raw_demo)} sygnałów EKG (po 1 z każdej klasy)")
"""),

md("""## 2. Wykres 12-odprowadzeniowy EKG dla każdej klasy"""),

code("""\
for i, cls in enumerate(SUPERCLASSES):
    fig = plot_ecg_12lead(
        X_raw_demo[i], fs=FS,
        title=f'Klasa: {cls} | {demo_df.iloc[i].get("report", "")}',
        color=CLASS_COLORS[cls],
        save_path=f'../results/viz_ecg_12lead_{cls}.png'
    )
    plt.show()
    print(f"  Zapisano: results/viz_ecg_12lead_{cls}.png")
"""),

md("""## 3. Porównanie klas – jedno odprowadzenie"""),

code("""\
# Wszystkie klasy na jednym wykresie, odprowadzenie II
signals_by_class = {cls: X_raw_demo[i] for i, cls in enumerate(SUPERCLASSES)}

fig = plot_ecg_comparison(
    signals_by_class, lead_idx=1, fs=FS,
    save_path='../results/viz_ecg_class_comparison.png'
)
plt.show()
print("Zapisano: results/viz_ecg_class_comparison.png")
"""),

md("""## 4. Rozkład klas"""),

code("""\
label_counts = Y['label_single'].value_counts().reindex(SUPERCLASSES)
fig = plot_class_distribution(
    label_counts,
    save_path='../results/viz_class_distribution.png'
)
plt.show()
"""),

md("""## 5. Potok przetwarzania wstępnego"""),

code("""\
# Użyj sygnału klasy NORM jako przykładu
example_raw = X_raw_demo[SUPERCLASSES.index('NORM')]

fig = plot_preprocessing_steps(
    example_raw, fs=FS, lead_idx=1,
    save_path='../results/viz_preprocessing_steps.png'
)
plt.show()
"""),

md("""## 6. Widma częstotliwościowe"""),

code("""\
ecg_filtered = bandpass_filter(
    notch_filter(remove_baseline_wander(example_raw, fs=FS), fs=FS),
    fs=FS
)
fig = plot_spectra_comparison(
    example_raw, ecg_filtered, fs=FS, lead_idx=1,
    save_path='../results/viz_spectra_comparison.png'
)
plt.show()
"""),

md("""## 7. Porównanie metod normalizacji"""),

code("""\
fig = plot_normalization_comparison(
    ecg_filtered, fs=FS, lead_idx=1,
    save_path='../results/viz_normalization_comparison.png'
)
plt.show()
"""),

md("""## 8. Pochodne sygnału"""),

code("""\
ecg_processed = preprocess_pipeline(example_raw, fs=FS)
fig = plot_derivatives(
    ecg_processed, fs=FS, lead_idx=1,
    save_path='../results/viz_derivatives.png'
)
plt.show()
"""),

md("""## 9. Dashboard oceny jakości

Obliczamy metryki jakości dla 50 przykładowych sygnałów.
"""),

code("""\
# Wczytaj więcej sygnałów do oceny jakości
Y_qual = pd.concat([
    Y[(Y.label_single == cls) & Y.index.isin(train_idx)].head(10)
    for cls in SUPERCLASSES
])
X_qual_raw = load_raw_data(Y_qual, sampling_rate=FS, data_path=DATA_PATH)

quality_list = [assess_signal_quality(ecg, fs=FS) for ecg in X_qual_raw]
df_quality = pd.DataFrame(quality_list)
df_quality['class'] = Y_qual['label_single'].values

fig = plot_quality_dashboard(
    df_quality,
    save_path='../results/viz_quality_dashboard.png'
)
plt.show()
"""),

md("""## 10. Porównanie modeli ML

> Uruchom najpierw Notebook 03, żeby wygenerować wyniki modeli.
> Poniżej re-trenujemy szybko na małym zbiorze demo.
"""),

code("""\
from sklearn.preprocessing import LabelEncoder
from src.feature_extraction import extract_features_batch
from src.classical_models import evaluate_all_models

# Mały zbiór do szybkiej demonstracji (20 próbek/klasę)
N = 20
Y_tr = pd.concat([Y[(Y.label_single == cls) & Y.index.isin(train_idx)].head(N) for cls in SUPERCLASSES])
Y_te = pd.concat([Y[(Y.label_single == cls) & ~Y.index.isin(train_idx) & ~Y.index.isin(train_idx)].head(N//4) for cls in SUPERCLASSES])
_, val_idx, test_idx = get_train_val_test_split(Y)
Y_te = pd.concat([Y[(Y.label_single == cls) & Y.index.isin(test_idx)].head(N//4) for cls in SUPERCLASSES])

X_tr_raw = load_raw_data(Y_tr, FS, DATA_PATH)
X_te_raw = load_raw_data(Y_te, FS, DATA_PATH)
X_tr = extract_features_batch(preprocess_batch(X_tr_raw, FS, verbose=False), FS, verbose=False)
X_te = extract_features_batch(preprocess_batch(X_te_raw, FS, verbose=False), FS, verbose=False)
X_tr = np.nan_to_num(X_tr); X_te = np.nan_to_num(X_te)

le = LabelEncoder(); le.fit(SUPERCLASSES)
y_tr = le.transform(Y_tr['label_single'].values)
y_te = le.transform(Y_te['label_single'].values)

summary_df, all_results = evaluate_all_models(X_tr, y_tr, X_te, y_te, classes=list(le.classes_))
"""),

code("""\
# Wykres porównawczy modeli
fig = plot_model_comparison(
    summary_df,
    save_path='../results/viz_model_comparison.png'
)
plt.show()
"""),

code("""\
# Macierze pomyłek
cm_results = {
    name: {
        'confusion_matrix': r['confusion_matrix'],
        'accuracy': r['accuracy'],
        'f1_macro': r['f1_macro'],
    }
    for name, r in all_results.items()
}
fig = plot_confusion_matrices(
    cm_results, list(le.classes_),
    title='Macierze pomyłek – Klasyczne metody ML',
    save_path='../results/viz_confusion_matrices_ml.png'
)
plt.show()
"""),

code("""\
# Feature importance (Random Forest)
from src.feature_extraction import get_feature_names

rf_model = all_results['Random Forest']['model']
importances = rf_model.feature_importances_
feature_names = get_feature_names()

fig = plot_feature_importance(
    importances, feature_names, top_n=20,
    save_path='../results/viz_feature_importance.png'
)
plt.show()
"""),

code("""\
# Heatmapa F1 per klasa
per_class_results = {
    name: {'y_true': le.inverse_transform(all_results[name]['y_pred']),
           'y_pred': le.inverse_transform(all_results[name]['y_pred'])}
    for name in all_results
}
# Poprawka: używamy prawdziwych etykiet
per_class_results = {
    name: {'y_true': le.inverse_transform(y_te),
           'y_pred': le.inverse_transform(r['y_pred'])}
    for name, r in all_results.items()
}

fig = plot_per_class_metrics(
    per_class_results, list(le.classes_),
    metric='f1-score',
    save_path='../results/viz_f1_heatmap.png'
)
plt.show()
"""),

md("""## 11. Podsumowanie – wszystkie wygenerowane pliki"""),

code("""\
import glob
result_files = sorted(glob.glob('../results/viz_*.png'))
print(f"Wygenerowano {len(result_files)} plików wizualizacji:")
for f in result_files:
    size_kb = os.path.getsize(f) // 1024
    print(f"  {os.path.basename(f):50s}  {size_kb:4d} KB")
"""),

]

save("05_visualizations.ipynb", viz_cells)

print("\\n✓ Wszystkie notebooki zostały wygenerowane w katalogu notebooks/")
