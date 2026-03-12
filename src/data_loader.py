"""
Moduł do ładowania i przygotowania zbioru danych PTB-XL.

PTB-XL to duży zbiór publicznych danych EKG zawierający:
- 21 837 klinicznych 12-odprowadzeniowych EKG od 18 885 pacjentów
- Sygnały w dwóch rozdzielczościach: 100 Hz i 500 Hz
- Etykiety diagnostyczne zakodowane w systemie SCP-ECG
- Podział na 10 foldów do walidacji krzyżowej (strat_fold)
"""

import os
import ast
import numpy as np
import pandas as pd
import wfdb


# ─────────────────────────────────────────────────────────────────────────────
# Superklasy diagnostyczne PTB-XL (5 klas)
# ─────────────────────────────────────────────────────────────────────────────
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SUPERCLASS_LABELS = {cls: idx for idx, cls in enumerate(SUPERCLASSES)}


def load_ptbxl_metadata(data_path: str) -> pd.DataFrame:
    """
    Wczytuje plik metadanych PTB-XL (ptbxl_database.csv).

    Parametry
    ---------
    data_path : str
        Ścieżka do katalogu głównego PTB-XL (zawierającego ptbxl_database.csv).

    Zwraca
    ------
    pd.DataFrame
        DataFrame z metadanymi; kolumna `scp_codes` sparsowana jako dict.
    """
    csv_path = os.path.join(data_path, 'ptbxl_database.csv')
    Y = pd.read_csv(csv_path, index_col='ecg_id')

    # scp_codes jest zapisany jako string reprezentujący słownik Python;
    # ast.literal_eval bezpiecznie go parsuje do prawdziwego dict
    Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
    return Y


def load_scp_statements(data_path: str) -> pd.DataFrame:
    """
    Wczytuje plik opisów kodów SCP (scp_statements.csv).

    Zawiera mapowanie kodu SCP → klasa diagnostyczna (NORM/MI/STTC/CD/HYP).

    Parametry
    ---------
    data_path : str
        Ścieżka do katalogu głównego PTB-XL.

    Zwraca
    ------
    pd.DataFrame
        DataFrame z opisami SCP, indeksowany kodem.
    """
    csv_path = os.path.join(data_path, 'scp_statements.csv')
    agg_df = pd.read_csv(csv_path, index_col=0)
    return agg_df


def aggregate_diagnostic(y_dic: dict, agg_df: pd.DataFrame) -> list:
    """
    Agreguje słownik kodów SCP do listy superklasowych etykiet diagnostycznych.

    Przykład: {'NORM': 100} → ['NORM']
              {'MI': 100, 'ISCAL': 70} → ['MI']  (ISCAL należy do MI)

    Parametry
    ---------
    y_dic : dict
        Słownik {kod_SCP: pewność} dla jednego rekordu EKG.
    agg_df : pd.DataFrame
        Tablica opisów SCP (scp_statements.csv) filtrowana do diagnostic==1.

    Zwraca
    ------
    list
        Lista unikalnych superklasowych etykiet diagnostycznych.
    """
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            c = agg_df.loc[key].diagnostic_class
            if str(c) != 'nan':
                tmp.append(c)
    return list(set(tmp))


def build_labels(Y: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """
    Buduje kolumny etykiet dla całego zbioru na podstawie kodów SCP.

    Tworzy dwie kolumny:
    - `diagnostic_superclass` : lista superklasowych etykiet (multi-label)
    - `label_single`          : pierwsza etykieta (single-label, do uproszczonej klasyfikacji)

    Parametry
    ---------
    Y : pd.DataFrame
        Metadane PTB-XL z kolumną scp_codes.
    data_path : str
        Ścieżka do katalogu PTB-XL (do wczytania scp_statements.csv).

    Zwraca
    ------
    pd.DataFrame
        DataFrame z dodanymi kolumnami etykiet.
    """
    agg_df = load_scp_statements(data_path)
    # Filtrujemy tylko kody oznaczone jako diagnostyczne
    agg_df_diag = agg_df[agg_df.diagnostic == 1]

    Y = Y.copy()
    Y['diagnostic_superclass'] = Y.scp_codes.apply(
        lambda x: aggregate_diagnostic(x, agg_df_diag)
    )

    # Odfiltruj rekordy bez etykiety (np. nieznane kody)
    Y = Y[Y['diagnostic_superclass'].map(len) > 0]

    # Single-label: bierzemy pierwszą superklasę (upraszcza klasyfikację)
    Y['label_single'] = Y['diagnostic_superclass'].apply(lambda x: x[0])

    return Y


def load_raw_data(df: pd.DataFrame, sampling_rate: int, data_path: str) -> np.ndarray:
    """
    Wczytuje surowe sygnały EKG z plików binarnych WFDB.

    Parametry
    ---------
    df : pd.DataFrame
        Fragment metadanych (wiersze do wczytania).
    sampling_rate : int
        Częstotliwość próbkowania: 100 (lr) lub 500 (hr).
    data_path : str
        Ścieżka do katalogu głównego PTB-XL.

    Zwraca
    ------
    np.ndarray
        Tablica kształtu (N, długość_sygnału, 12) z sygnałami EKG.
    """
    if sampling_rate == 100:
        filenames = df.filename_lr
    else:
        filenames = df.filename_hr

    data = []
    for fname in filenames:
        full_path = os.path.join(data_path, fname)
        record = wfdb.rdsamp(full_path)
        data.append(record[0])   # record[0] = sygnał (ndarray), record[1] = metadata

    return np.array(data)


def get_train_val_test_split(Y: pd.DataFrame):
    """
    Zwraca indeksy dla podziału train/val/test zgodnie z PTB-XL strat_fold.

    Standardowy podział:
    - Trening : fold 1–8
    - Walidacja: fold 9
    - Test     : fold 10

    Parametry
    ---------
    Y : pd.DataFrame
        Metadane z kolumną `strat_fold`.

    Zwraca
    ------
    tuple
        (train_idx, val_idx, test_idx) – indeksy ecg_id.
    """
    train_idx = Y[Y.strat_fold <= 8].index
    val_idx   = Y[Y.strat_fold == 9].index
    test_idx  = Y[Y.strat_fold == 10].index
    return train_idx, val_idx, test_idx
