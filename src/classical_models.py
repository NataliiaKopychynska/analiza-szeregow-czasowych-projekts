"""
Klasyczne metody uczenia maszynowego do klasyfikacji EKG.

Zaimplementowane algorytmy (≥4 zgodnie z wymaganiami projektu):
1. Regresja logistyczna (Logistic Regression)
2. Las losowy (Random Forest)
3. Maszyna wektorów nośnych z jądrem RBF (SVM)
4. k-Nearest Neighbors (KNN)
5. Gradient Boosting (XGBoost-like, sklearn)
6. Naiwny klasyfikator Bayesa (Gaussian Naive Bayes) – model bazowy

Każdy model opakowany jest w Pipeline z preprocessing (StandardScaler),
gdzie to wymagane.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
import numpy as np
import pandas as pd
import time


# ─────────────────────────────────────────────────────────────────────────────
# Definicje modeli
# ─────────────────────────────────────────────────────────────────────────────

def get_classical_models() -> dict:
    """
    Zwraca słownik nieoptymalizowanych modeli klasycznych.

    Modele używają domyślnych (lub minimalnie sensownych) hiperparametrów
    – bez strojenia, zgodnie z wymaganiami Kontroli 2.

    Zwraca
    ------
    dict
        Słownik {nazwa_modelu: obiekt_modelu}.
    """
    models = {

        # ── 1. Regresja Logistyczna ──────────────────────────────────────────
        # Model liniowy maksymalizujący wiarygodność logarytmiczną.
        # Regularyzacja L2 (domyślna) zapobiega nadmiernemu dopasowaniu.
        # Pipeline ze StandardScaler: LR jest wrażliwa na skalę cech.
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                multi_class='multinomial',  # klasyfikacja wieloklasowa softmax
                solver='lbfgs',             # wydajny solver dla małych zbiorów
                max_iter=1000,
                random_state=42
            ))
        ]),

        # ── 2. Las Losowy ────────────────────────────────────────────────────
        # Ensemble drzew decyzyjnych uczonych na losowych podzbiorach cech.
        # Odporny na wartości odstające; nie wymaga skalowania cech.
        # n_estimators=100 – kompromis szybkość/jakość.
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=None,       # drzewa rosną do czystych liści
            n_jobs=-1,            # równoległe obliczenia na wszystkich CPU
            random_state=42
        ),

        # ── 3. SVM z jądrem RBF ──────────────────────────────────────────────
        # Maszyna wektorów nośnych z jądrem radialnym – efektywna dla
        # nieliniowych granic decyzyjnych w przestrzeni cech.
        # WYMAGA skalowania cech (StandardScaler).
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                C=1.0,           # parametr regularyzacji
                gamma='scale',   # γ = 1/(n_features * var(X))
                probability=True,  # włącza predict_proba (potrzebne do ROC)
                random_state=42
            ))
        ]),

        # ── 4. k-Nearest Neighbors ───────────────────────────────────────────
        # Klasyfikacja przez głosowanie k najbliższych sąsiadów w przestrzeni cech.
        # k=5 – standardowy wybór, odporny na szum.
        # WYMAGA skalowania (odległości euklidesowe wrażliwe na skalę).
        'KNN (k=5)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean',
                n_jobs=-1
            ))
        ]),

        # ── 5. Gradient Boosting ────────────────────────────────────────────
        # Sekwencyjny ensemble słabych klasyfikatorów (drzew decyzyjnych).
        # Skuteczniejszy od RF dla wielu zadań, ale wolniejszy w treningu.
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        # ── 6. Naiwny Bayes (model bazowy) ──────────────────────────────────
        # Probabilistyczny model zakładający niezależność cech.
        # Bardzo szybki – służy jako punkt odniesienia (baseline).
        'Naive Bayes': GaussianNB(),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Funkcje treningowe i ewaluacyjne
# ─────────────────────────────────────────────────────────────────────────────

def train_evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = '',
    classes: list = None
) -> dict:
    """
    Trenuje model i oblicza metryki na zbiorze testowym.

    Parametry
    ---------
    model : sklearn estimator
        Model do trenowania (pipeline lub klasyfikator).
    X_train, y_train : np.ndarray
        Dane treningowe.
    X_test, y_test : np.ndarray
        Dane testowe.
    model_name : str
        Nazwa modelu (do wydruku).
    classes : list
        Lista nazw klas (do raportu).

    Zwraca
    ------
    dict
        Słownik z modelem, predykcjami i metrykami.
    """
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # ── Trening ──
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Czas treningu: {train_time:.1f}s")

    # ── Predykcja ──
    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    # ── Metryki ──
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  Dokładność (Accuracy):         {acc:.4f}")
    print(f"  Zbalansowana dokładność:       {balanced_acc:.4f}")
    print(f"  F1-score (macro):              {f1_macro:.4f}")
    print(f"  F1-score (weighted):           {f1_weighted:.4f}")
    print(f"  Czas predykcji:                {pred_time:.3f}s")
    print()
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    return {
        'model': model,
        'name': model_name,
        'y_pred': y_pred,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'train_time': train_time,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }


def evaluate_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: list = None
) -> pd.DataFrame:
    """
    Trenuje i ewaluuje wszystkie klasyczne modele, zwraca zbiorczy DataFrame.

    Parametry
    ---------
    X_train, y_train, X_test, y_test : np.ndarray
        Dane treningowe i testowe.
    classes : list
        Nazwy klas.

    Zwraca
    ------
    pd.DataFrame
        Tabela porównawcza wyników wszystkich modeli.
    """
    models = get_classical_models()
    all_results = {}

    for name, model in models.items():
        result = train_evaluate_model(
            model, X_train, y_train, X_test, y_test,
            model_name=name, classes=classes
        )
        all_results[name] = result

    # Podsumowanie w DataFrame
    summary = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': r['accuracy'],
            'Balanced Accuracy': r['balanced_accuracy'],
            'F1 Macro': r['f1_macro'],
            'F1 Weighted': r['f1_weighted'],
            'Train Time [s]': round(r['train_time'], 2),
        }
        for name, r in all_results.items()
    ])
    summary = summary.sort_values('F1 Macro', ascending=False).reset_index(drop=True)

    return summary, all_results
