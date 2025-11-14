"""
Training Script für Rechnungsklassifizierung
Trainiert verschiedene ML-Models und wählt das beste aus
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import InvoiceDataPreprocessor


class InvoiceClassifierTrainer:
    """
    Trainer für verschiedene Klassifizierungs-Models
    """

    def __init__(self, preprocessor):
        """
        Args:
            preprocessor: Gefittete InvoiceDataPreprocessor Instanz
        """
        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0

    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """
        Trainiert Logistic Regression (Baseline)
        """
        print("\n" + "="*60)
        print("Training: Logistic Regression (Baseline)")
        print("="*60)

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        print("Fitting model...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)

        # Metriken
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_macro = f1_score(y_val, y_pred, average='macro')

        print(f"\n✓ Training abgeschlossen!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")

        # Speichern
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        }

        # Best model tracking
        if f1_weighted > self.best_score:
            self.best_score = f1_weighted
            self.best_model = model
            self.best_model_name = 'logistic_regression'

        return model, accuracy, f1_weighted

    def train_random_forest(self, X_train, y_train, X_val, y_val,
                           n_estimators=100, max_depth=None):
        """
        Trainiert Random Forest Classifier
        """
        print("\n" + "="*60)
        print("Training: Random Forest Classifier")
        print("="*60)
        print(f"Parameters: n_estimators={n_estimators}, max_depth={max_depth}")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        print("Fitting model...")
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)

        # Metriken
        accuracy = accuracy_score(y_val, y_pred)
        f1_weighted = f1_score(y_val, y_pred, average='weighted')
        f1_macro = f1_score(y_val, y_pred, average='macro')

        print(f"\n✓ Training abgeschlossen!")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1_weighted:.4f}")
        print(f"  F1-Score (macro): {f1_macro:.4f}")

        # Feature Importance (Top 10)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print(f"\n  Top 10 Feature Importances:")
            for idx in top_indices:
                print(f"    Feature {idx}: {importances[idx]:.4f}")

        # Speichern
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'n_estimators': n_estimators,
            'max_depth': max_depth
        }

        # Best model tracking
        if f1_weighted > self.best_score:
            self.best_score = f1_weighted
            self.best_model = model
            self.best_model_name = 'random_forest'

        return model, accuracy, f1_weighted

    def print_detailed_report(self, model, X_val, y_val, model_name):
        """
        Druckt detaillierten Classification Report
        """
        print(f"\n{'='*60}")
        print(f"Detailed Report: {model_name}")
        print(f"{'='*60}")

        y_pred = model.predict(X_val)

        # Classification Report - nur für Klassen die im Validation Set vorkommen
        unique_labels = np.unique(np.concatenate([y_val, y_pred]))
        label_names = [self.preprocessor.category_encoder.classes_[i] for i in unique_labels]

        report = classification_report(
            y_val, y_pred,
            labels=unique_labels,
            target_names=label_names,
            digits=4,
            zero_division=0
        )
        print("\nClassification Report:")
        print(report)

    def compare_models(self):
        """
        Vergleicht alle trainierten Models
        """
        print("\n" + "="*60)
        print("Model Vergleich")
        print("="*60)

        results_df = pd.DataFrame(self.results).T
        print(results_df.to_string())

        print(f"\n🏆 Bestes Model: {self.best_model_name}")
        print(f"   F1-Score (weighted): {self.best_score:.4f}")

    def save_best_model(self, save_path):
        """
        Speichert das beste Model
        """
        if self.best_model is None:
            raise ValueError("Kein Model trainiert!")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'best_score': self.best_score,
            'results': self.results
        }

        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n✓ Bestes Model gespeichert: {save_path}")
        print(f"  Model: {self.best_model_name}")
        print(f"  F1-Score: {self.best_score:.4f}")

    @classmethod
    def load_model(cls, load_path):
        """
        Lädt ein gespeichertes Model
        """
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)

        print(f"✓ Model geladen: {load_path}")
        print(f"  Model: {model_data['model_name']}")
        print(f"  F1-Score: {model_data['best_score']:.4f}")

        return model_data['model'], model_data['model_name']


def main():
    """
    Haupt-Training Pipeline
    """
    print("="*60)
    print("RECHNUNGS-KLASSIFIZIERUNG: MODEL TRAINING")
    print("="*60)

    # 1. Daten vorbereiten
    print("\n[1/4] Lade und bereite Daten vor...")
    train_csv = "../Data/Dataset/Train.csv"

    preprocessor = InvoiceDataPreprocessor(
        max_features=5000,
        test_size=0.2,
        random_state=42
    )

    X_train, X_val, y_train, y_val, label_mapping = preprocessor.prepare_train_data(train_csv)

    # Pipeline speichern
    preprocessor.save_pipeline("models/preprocessing_pipeline.pkl")

    # 2. Trainer initialisieren
    print("\n[2/4] Initialisiere Trainer...")
    trainer = InvoiceClassifierTrainer(preprocessor)

    # 3. Models trainieren
    print("\n[3/4] Trainiere Models...")

    # Logistic Regression (schnell, baseline)
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)

    # Random Forest (besser, aber langsamer)
    trainer.train_random_forest(
        X_train, y_train, X_val, y_val,
        n_estimators=100,
        max_depth=20
    )

    # Optional: Zweites Random Forest mit anderen Parametern
    # trainer.train_random_forest(
    #     X_train, y_train, X_val, y_val,
    #     n_estimators=200,
    #     max_depth=30
    # )

    # 4. Vergleichen und Speichern
    print("\n[4/4] Vergleiche Models und speichere bestes...")
    trainer.compare_models()

    # Detaillierter Report für bestes Model
    trainer.print_detailed_report(
        trainer.best_model,
        X_val, y_val,
        trainer.best_model_name
    )

    # Bestes Model speichern
    trainer.save_best_model("models/classifier_model.pkl")

    # Label Mapping speichern
    label_mapping_path = Path("models/label_mapping.json")
    label_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    print(f"\n✓ Label Mapping gespeichert: {label_mapping_path}")

    print("\n" + "="*60)
    print("✓ TRAINING ABGESCHLOSSEN!")
    print("="*60)
    print("\nGespeicherte Dateien:")
    print("  - models/preprocessing_pipeline.pkl")
    print("  - models/classifier_model.pkl")
    print("  - models/label_mapping.json")
    print("\nNächster Schritt: classify_invoice.py für Predictions")


if __name__ == "__main__":
    main()
