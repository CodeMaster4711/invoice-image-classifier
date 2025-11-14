"""
Datenvorbereitungs-Script für Rechnungsklassifizierung
Bereitet Rohdaten auf und erstellt Features für ML-Models
"""

import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class InvoiceDataPreprocessor:
    """
    Haupt-Klasse für Datenvorverarbeitung von Rechnungsdaten
    """

    def __init__(self, max_features=5000, test_size=0.2, random_state=42):
        """
        Args:
            max_features: Maximale Anzahl TF-IDF Features
            test_size: Anteil der Validierungsdaten (0-1)
            random_state: Seed für Reproduzierbarkeit
        """
        self.max_features = max_features
        self.test_size = test_size
        self.random_state = random_state

        # Encoders und Transformers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.vendor_encoder = LabelEncoder()
        self.gl_encoder = LabelEncoder()
        self.amount_scaler = StandardScaler()
        self.category_encoder = LabelEncoder()

        # Fitted Status
        self.is_fitted = False

    def clean_text(self, text):
        """
        Bereinigt Textdaten

        Args:
            text: Roher Text

        Returns:
            Bereinigter Text
        """
        if pd.isna(text):
            return ""

        # Zu String konvertieren
        text = str(text)

        # Lowercase
        text = text.lower()

        # Sonderzeichen entfernen (aber Leerzeichen behalten)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Multiple Leerzeichen zu einem
        text = re.sub(r'\s+', ' ', text)

        # Trim
        text = text.strip()

        return text

    def preprocess_features(self, df, fit=False):
        """
        Verarbeitet alle Features

        Args:
            df: DataFrame mit Rechnungsdaten
            fit: Ob Transformer gefittet werden sollen (True für Training)

        Returns:
            Feature-Matrix (numpy array)
        """
        # Kopie erstellen
        data = df.copy()

        # 1. Text-Feature (Item_Description)
        data['cleaned_description'] = data['Item_Description'].apply(self.clean_text)

        if fit:
            text_features = self.tfidf_vectorizer.fit_transform(data['cleaned_description'])
        else:
            text_features = self.tfidf_vectorizer.transform(data['cleaned_description'])

        # 2. Vendor_Code Feature
        if fit:
            vendor_features = self.vendor_encoder.fit_transform(data['Vendor_Code'])
        else:
            # Bei neuen Vendors: auf "unknown" mappen
            known_vendors = set(self.vendor_encoder.classes_)
            data['Vendor_Code_mapped'] = data['Vendor_Code'].apply(
                lambda x: x if x in known_vendors else self.vendor_encoder.classes_[0]
            )
            vendor_features = self.vendor_encoder.transform(data['Vendor_Code_mapped'])

        vendor_features = vendor_features.reshape(-1, 1)

        # 3. GL_Code Feature
        if fit:
            gl_features = self.gl_encoder.fit_transform(data['GL_Code'])
        else:
            known_gl = set(self.gl_encoder.classes_)
            data['GL_Code_mapped'] = data['GL_Code'].apply(
                lambda x: x if x in known_gl else self.gl_encoder.classes_[0]
            )
            gl_features = self.gl_encoder.transform(data['GL_Code_mapped'])

        gl_features = gl_features.reshape(-1, 1)

        # 4. Invoice Amount Feature
        amount_features = data['Inv_Amt'].values.reshape(-1, 1)

        if fit:
            amount_features = self.amount_scaler.fit_transform(amount_features)
        else:
            amount_features = self.amount_scaler.transform(amount_features)

        # Alle Features zusammenführen
        # TF-IDF ist sparse, Rest ist dense
        from scipy.sparse import hstack, csr_matrix

        features = hstack([
            text_features,
            csr_matrix(vendor_features),
            csr_matrix(gl_features),
            csr_matrix(amount_features)
        ])

        return features

    def prepare_train_data(self, train_csv_path):
        """
        Bereitet Trainingsdaten vor und splittet in Train/Validation

        Args:
            train_csv_path: Pfad zur Train.csv Datei

        Returns:
            X_train, X_val, y_train, y_val, label_mapping
        """
        print(f"Lade Trainingsdaten von {train_csv_path}...")
        df = pd.read_csv(train_csv_path)

        print(f"Datensatz: {len(df)} Zeilen, {len(df.columns)} Spalten")
        print(f"Kategorien: {df['Product_Category'].nunique()} verschiedene Klassen")

        # Null-Werte prüfen
        null_counts = df.isnull().sum()
        if null_counts.any():
            print("\nWarnung: Null-Werte gefunden:")
            print(null_counts[null_counts > 0])
            print("Fülle Null-Werte auf...")
            df = df.fillna({
                'Item_Description': '',
                'Vendor_Code': 'VENDOR-UNKNOWN',
                'GL_Code': 'GL-UNKNOWN',
                'Inv_Amt': 0.0
            })

        # Features vorbereiten
        print("\nBereite Features vor...")
        X = self.preprocess_features(df, fit=True)

        # Labels enkodieren
        print("Enkodiere Labels...")
        y = self.category_encoder.fit_transform(df['Product_Category'])

        # Train/Val Split
        print(f"\nSplitte Daten (Test-Size: {self.test_size})...")

        # Prüfe Klassenverteilung
        unique, counts = np.unique(y, return_counts=True)
        min_samples = counts.min()

        # Nur stratify wenn alle Klassen mindestens 2 Samples haben
        stratify_param = y if min_samples >= 2 else None

        if stratify_param is None:
            print(f"Warnung: {sum(counts < 2)} Klassen haben < 2 Samples. Stratify deaktiviert.")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print(f"Training Set: {X_train.shape[0]} Samples")
        print(f"Validation Set: {X_val.shape[0]} Samples")
        print(f"Feature Dimensionen: {X_train.shape[1]}")

        self.is_fitted = True

        # Label Mapping für spätere Verwendung
        label_mapping = {
            idx: label
            for idx, label in enumerate(self.category_encoder.classes_)
        }

        return X_train, X_val, y_train, y_val, label_mapping

    def prepare_test_data(self, test_csv_path):
        """
        Bereitet Test-Daten vor (ohne Labels)

        Args:
            test_csv_path: Pfad zur Test.csv Datei

        Returns:
            X_test, test_ids
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor muss erst auf Trainingsdaten gefittet werden!")

        print(f"Lade Test-Daten von {test_csv_path}...")
        df = pd.read_csv(test_csv_path)

        print(f"Test-Datensatz: {len(df)} Zeilen")

        # Null-Werte auffüllen
        df = df.fillna({
            'Item_Description': '',
            'Vendor_Code': 'VENDOR-UNKNOWN',
            'GL_Code': 'GL-UNKNOWN',
            'Inv_Amt': 0.0
        })

        # Features vorbereiten (ohne fit!)
        X_test = self.preprocess_features(df, fit=False)
        test_ids = df['Inv_Id'].values

        print(f"Test Features: {X_test.shape}")

        return X_test, test_ids

    def save_pipeline(self, save_path):
        """
        Speichert die komplette Preprocessing-Pipeline

        Args:
            save_path: Pfad zum Speichern (z.B. 'models/preprocessing_pipeline.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor muss erst gefittet werden!")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pipeline_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'vendor_encoder': self.vendor_encoder,
            'gl_encoder': self.gl_encoder,
            'amount_scaler': self.amount_scaler,
            'category_encoder': self.category_encoder,
            'max_features': self.max_features,
            'test_size': self.test_size,
            'random_state': self.random_state
        }

        with open(save_path, 'wb') as f:
            pickle.dump(pipeline_data, f)

        print(f"\nPipeline gespeichert: {save_path}")

    @classmethod
    def load_pipeline(cls, load_path):
        """
        Lädt eine gespeicherte Pipeline

        Args:
            load_path: Pfad zur gespeicherten Pipeline

        Returns:
            InvoiceDataPreprocessor Instanz
        """
        with open(load_path, 'rb') as f:
            pipeline_data = pickle.load(f)

        # Neue Instanz erstellen
        preprocessor = cls(
            max_features=pipeline_data['max_features'],
            test_size=pipeline_data['test_size'],
            random_state=pipeline_data['random_state']
        )

        # Transformer laden
        preprocessor.tfidf_vectorizer = pipeline_data['tfidf_vectorizer']
        preprocessor.vendor_encoder = pipeline_data['vendor_encoder']
        preprocessor.gl_encoder = pipeline_data['gl_encoder']
        preprocessor.amount_scaler = pipeline_data['amount_scaler']
        preprocessor.category_encoder = pipeline_data['category_encoder']
        preprocessor.is_fitted = True

        print(f"Pipeline geladen von: {load_path}")

        return preprocessor


def main():
    """
    Beispiel-Verwendung
    """
    # Pfade
    train_csv = "../Data/Dataset/Train.csv"

    # Preprocessor erstellen
    preprocessor = InvoiceDataPreprocessor(
        max_features=5000,
        test_size=0.2,
        random_state=42
    )

    # Trainingsdaten vorbereiten
    X_train, X_val, y_train, y_val, label_mapping = preprocessor.prepare_train_data(train_csv)

    # Pipeline speichern
    preprocessor.save_pipeline("models/preprocessing_pipeline.pkl")

    print("\n=== Zusammenfassung ===")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Validation Samples: {X_val.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Kategorien: {len(label_mapping)}")
    print(f"\nBeispiel Kategorien: {list(label_mapping.values())[:5]}")


if __name__ == "__main__":
    main()
