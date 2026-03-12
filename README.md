# Automatyczna klasyfikacja sygnałów EKG – PTB-XL

Projekt implementuje i porównuje algorytmy uczenia maszynowego do klasyfikacji
wielokanałowych sygnałów EKG ze zbioru PTB-XL.

---

## Struktura katalogów

```
analiza-szeregow-czasowych-projekts/
│
├── data/                          ← TU UMIEŚĆ POBRANY ZBIÓR PTB-XL
│   ├── ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
│   │   ├── ptbxl_database.csv
│   │   ├── scp_statements.csv
│   │   ├── records100/            ← sygnały 100 Hz
│   │   └── records500/            ← sygnały 500 Hz
│   └── processed/                 ← generowane automatycznie (Notebook 2)
│
├── notebooks/                     ← Jupyter Notebooks (uruchamiaj po kolei)
│   ├── 01_eda.ipynb               ← Eksploracyjna Analiza Danych (Kontrola 1)
│   ├── 02_preprocessing.ipynb     ← Potok przetwarzania + ocena jakości
│   ├── 03_classical_ml.ipynb      ← Klasyczne metody ML (LR, RF, SVM, KNN, GB)
│   └── 04_deep_learning.ipynb     ← Głębokie sieci (CNN1D, ResNet1D, BiLSTM)
│
├── src/                           ← Moduły źródłowe Python
│   ├── data_loader.py             ← Wczytywanie PTB-XL, etykiety SCP
│   ├── preprocessing.py           ← Filtracja, normalizacja, ocena jakości
│   ├── feature_extraction.py      ← Ekstrakcja 228 cech statystycznych
│   ├── classical_models.py        ← 6 klasycznych algorytmów ML
│   └── deep_models.py             ← CNN1D, ResNet1D, BiLSTM (PyTorch)
│
├── models/                        ← Zapisywane wytrenowane modele
├── results/                       ← Generowane wykresy (PNG)
│
├── requirements.txt               ← Wymagane biblioteki Python
├── create_notebooks.py            ← Skrypt generujący notebooki (uruchomiony raz)
└── README.md                      ← Instrukcja obsługi
```

---

## Wymagania wstępne

- Python **3.9+**
- pip lub conda
- ~3 GB wolnego miejsca na dysku (zbiór danych)
- GPU (opcjonalnie, przyspiesza Deep Learning; projekt działa też na CPU)

---

## Krok 1 – Pobranie zbioru danych PTB-XL

Zbiór danych jest publicznie dostępny na platformie PhysioNet:

### Opcja A – wget (Linux/macOS)
```bash
cd data/
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

### Opcja B – narzędzie physionet (wymaga rejestracji)
```bash
pip install wfdb
python -c "import wfdb; wfdb.dl_database('ptb-xl', dl_dir='data/ptb-xl-1.0.3')"
```

### Opcja C – ręczne pobranie
1. Przejdź na stronę: https://physionet.org/content/ptb-xl/1.0.3/
2. Zaloguj się (wymagane) lub pobierz jako gość
3. Pobierz archiwum ZIP i rozpakuj do katalogu `data/`

**Docelowa struktura:**
```
data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
    ptbxl_database.csv
    scp_statements.csv
    records100/
    records500/
```

---

## Krok 2 – Instalacja środowiska Python

```bash
# Utwórz wirtualne środowisko (zalecane)
python3 -m venv venv
source venv/bin/activate       # Linux/macOS
# lub: venv\Scripts\activate   # Windows

# Zainstaluj zależności
pip install -r requirements.txt
```

Weryfikacja instalacji:
```bash
python3 -c "import wfdb, torch, sklearn; print('OK')"
```

---

## Krok 3 – Uruchomienie notebooków Jupyter

```bash
# Uruchom Jupyter Lab
jupyter lab

# lub klasyczny Jupyter Notebook
jupyter notebook
```

Otwórz przeglądarkę pod adresem: `http://localhost:8888`

### Kolejność uruchamiania notebooków:

| Notebook | Opis | Kontrola |
|---|---|---|
| `01_eda.ipynb` | Eksploracyjna Analiza Danych | Kontrola 1 |
| `02_preprocessing.ipynb` | Potok przetwarzania + ocena jakości | **Kontrola 2** |
| `03_classical_ml.ipynb` | Klasyczne metody ML (6 algorytmów) | **Kontrola 2** |
| `04_deep_learning.ipynb` | Głębokie sieci neuronowe (3 architektury) | **Kontrola 2** |

**Ważne:** Uruchamiaj notebooki w kolejności 01 → 02 → 03 → 04.
Notebook 03 i 04 wczytują dane generowane przez Notebook 02.

---

## Krok 4 – Zmiana ścieżki do danych

W każdym notebooku znajduje się komórka z definicją `DATA_PATH`:

```python
DATA_PATH = '../data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3'
```

Jeśli pobrałeś dane do innej lokalizacji, zmień tę ścieżkę.

---

## Parametry do dostosowania

### Rozmiar podzbioru (szybkość vs. jakość)
W notebookach 03 i 04 możesz zmienić `N_PER_CLASS`:
```python
N_PER_CLASS = 100    # 100 próbek/klasę – szybka demonstracja (default)
N_PER_CLASS = None   # pełny zbiór – pełne wyniki (kilka godzin)
```

### Liczba epok (Notebook 04)
```python
EPOCHS = 20    # szybka demonstracja
EPOCHS = 50    # lepsze wyniki
```

---

## Wyniki

Po uruchomieniu notebooków w katalogu `results/` pojawią się pliki PNG:

- `eda_class_distribution.png` – rozkład klas diagnostycznych
- `eda_demographics.png` – demografia pacjentów
- `eda_ecg_*.png` – przykładowe sygnały EKG (po jednym z każdej klasy)
- `preprocessing_steps.png` – wizualizacja kroków przetwarzania
- `preprocessing_spectra.png` – widma częstotliwościowe przed/po filtracji
- `preprocessing_quality_histograms.png` – metryki jakości sygnałów
- `preprocessing_derivatives.png` – sygnał i jego pochodne
- `classical_ml_comparison.png` – porównanie modeli ML
- `classical_ml_confusion_matrices.png` – macierze pomyłek
- `classical_ml_feature_importance.png` – ważność cech (RF)
- `deep_learning_training_curves.png` – krzywe uczenia DL
- `deep_learning_confusion_matrices.png` – macierze pomyłek DL

---

## Opis metod

### Klasyczne ML (Notebook 03)
1. **Regresja Logistyczna** – liniowy klasyfikator softmax, regularyzacja L2
2. **Las Losowy** – 100 drzew, Gini impurity, równoległe obliczenia
3. **SVM (RBF)** – jądro radialną bazą, C=1.0, gamma='scale'
4. **KNN (k=5)** – 5 najbliższych sąsiadów, metryka euklidesowa
5. **Gradient Boosting** – 100 estymatorów, learning_rate=0.1
6. **Naiwny Bayes** – model bazowy (punkt odniesienia)

### Głębokie uczenie (Notebook 04)
1. **CNN1D** – 3 bloki Conv1d(+BN+ReLU+MaxPool) + 2 FC, Dropout 0.5
2. **ResNet1D** – 4 grupy bloków rezydualnych (64→128→256→512 kanałów)
3. **BiLSTM** – 2-warstwowy LSTM bidirectional + mechanizm uwagi (attention)

### Przetwarzanie wstępne (Notebook 02)
- Filtr pasmowoprzepustowy Butterwortha [0.5–40 Hz], rząd 4
- Filtr notch IIR 50 Hz (zakłócenia sieciowe)
- Usunięcie dryftu bazowego (filtr medianowy 200ms + 600ms)
- Normalizacja Z-score (kanał po kanale)
- Ekstrakcja 228 cech: statystyczne + energetyczne + spektralne + gradientowe

---

## Rozwiązywanie problemów

**Błąd `ModuleNotFoundError: No module named 'wfdb'`**
```bash
pip install wfdb
```

**Błąd `FileNotFoundError` przy DATA_PATH**
Sprawdź czy zbiór danych jest w odpowiednim katalogu i zmień `DATA_PATH`.

**Wolny trening (CPU)**
Zmniejsz `N_PER_CLASS = 50` i `EPOCHS = 10`.

**Brak pamięci RAM**
Zmniejsz `BATCH_SIZE = 16` w Notebook 04.
