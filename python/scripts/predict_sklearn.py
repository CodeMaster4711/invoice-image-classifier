"""
Klassifizierungs-Script für Rechnungen
Verwendet trainiertes Model für Predictions
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessing import InvoiceDataPreprocessor
from scripts.train_sklearn import InvoiceClassifierTrainer


class InvoiceClassifier:
    """
    Hauptklasse für Rechnungsklassifizierung
    """

    def __init__(self, model_dir="models"):
        """
        Args:
            model_dir: Verzeichnis mit gespeicherten Models
        """
        self.model_dir = Path(model_dir)

        # Lade Pipeline und Model
        print("Lade Models...")
        self.preprocessor = InvoiceDataPreprocessor.load_pipeline(
            self.model_dir / "preprocessing_pipeline.pkl"
        )

        self.model, self.model_name = InvoiceClassifierTrainer.load_model(
            self.model_dir / "classifier_model.pkl"
        )

        # Lade Label Mapping
        with open(self.model_dir / "label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            # Keys sind Strings aus JSON, zu int konvertieren
            self.label_mapping = {int(k): v for k, v in label_mapping.items()}

        print(f"✓ Initialisierung abgeschlossen!")
        print(f"  Model: {self.model_name}")
        print(f"  Kategorien: {len(self.label_mapping)}\n")

    def predict_single(self, vendor_code, gl_code, inv_amt, item_description):
        """
        Klassifiziert eine einzelne Rechnung

        Args:
            vendor_code: Vendor Code (z.B. "VENDOR-1676")
            gl_code: GL Code (z.B. "GL-6100410")
            inv_amt: Rechnungsbetrag (float)
            item_description: Textbeschreibung

        Returns:
            dict mit prediction, confidence, probabilities
        """
        # DataFrame erstellen
        data = pd.DataFrame([{
            'Vendor_Code': vendor_code,
            'GL_Code': gl_code,
            'Inv_Amt': inv_amt,
            'Item_Description': item_description
        }])

        # Features vorbereiten
        X = self.preprocessor.preprocess_features(data, fit=False)

        # Prediction
        y_pred = self.model.predict(X)[0]
        predicted_category = self.label_mapping[y_pred]

        # Confidence (Wahrscheinlichkeiten)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]

            # Top 3 Kategorien mit höchster Wahrscheinlichkeit
            top_3_idx = np.argsort(probs)[-3:][::-1]
            top_3_probs = [(self.label_mapping[idx], probs[idx]) for idx in top_3_idx]

            # Confidence ist die Wahrscheinlichkeit der Top-Prediction
            confidence = probs[top_3_idx[0]]
        else:
            confidence = None
            top_3_probs = None

        return {
            'predicted_category': predicted_category,
            'confidence': confidence,
            'top_3_predictions': top_3_probs
        }

    def predict_batch(self, csv_path):
        """
        Klassifiziert mehrere Rechnungen aus CSV

        Args:
            csv_path: Pfad zur CSV-Datei mit Rechnungen

        Returns:
            DataFrame mit Predictions
        """
        print(f"Lade Daten von {csv_path}...")
        df = pd.read_csv(csv_path)

        # Null-Werte auffüllen
        df = df.fillna({
            'Item_Description': '',
            'Vendor_Code': 'VENDOR-UNKNOWN',
            'GL_Code': 'GL-UNKNOWN',
            'Inv_Amt': 0.0
        })

        print(f"Klassifiziere {len(df)} Rechnungen...")

        # Features vorbereiten
        X = self.preprocessor.preprocess_features(df, fit=False)

        # Predictions
        y_pred = self.model.predict(X)
        predictions = [self.label_mapping[idx] for idx in y_pred]

        # Confidence (falls verfügbar)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)
            confidences = [probs[i, y_pred[i]] for i in range(len(y_pred))]
        else:
            confidences = [None] * len(y_pred)

        # Ergebnis DataFrame
        results = df.copy()
        results['Predicted_Category'] = predictions
        results['Confidence'] = confidences

        print(f"✓ Klassifizierung abgeschlossen!")

        return results

    def predict_and_save_submission(self, test_csv_path, output_path="submission.csv"):
        """
        Erstellt Submission-File für Kaggle/Competition

        Args:
            test_csv_path: Pfad zur Test.csv
            output_path: Pfad für Submission-File
        """
        print("Erstelle Submission File...")

        # Predictions
        results = self.predict_batch(test_csv_path)

        # Submission Format (nur ID und Kategorie)
        submission = pd.DataFrame({
            'Inv_Id': results['Inv_Id'],
            'Product_Category': results['Predicted_Category']
        })

        # Speichern
        submission.to_csv(output_path, index=False)
        print(f"✓ Submission gespeichert: {output_path}")
        print(f"  {len(submission)} Predictions")

        return submission


def interactive_mode(classifier):
    """
    Interaktiver Modus für einzelne Rechnungen
    """
    print("\n" + "="*60)
    print("INTERAKTIVER MODUS - Rechnungsklassifizierung")
    print("="*60)
    print("Gib die Rechnungsdaten ein (oder 'exit' zum Beenden)\n")

    while True:
        try:
            print("-" * 60)
            vendor_code = input("Vendor Code (z.B. VENDOR-1676): ").strip()
            if vendor_code.lower() == 'exit':
                break

            gl_code = input("GL Code (z.B. GL-6100410): ").strip()
            if gl_code.lower() == 'exit':
                break

            inv_amt = input("Rechnungsbetrag (z.B. 83.24): ").strip()
            if inv_amt.lower() == 'exit':
                break
            inv_amt = float(inv_amt)

            item_description = input("Item Description: ").strip()
            if item_description.lower() == 'exit':
                break

            # Prediction
            result = classifier.predict_single(
                vendor_code, gl_code, inv_amt, item_description
            )

            # Ausgabe
            print("\n" + "="*60)
            print("ERGEBNIS:")
            print("="*60)
            print(f"📊 Kategorie: {result['predicted_category']}")

            if result['confidence'] is not None:
                print(f"✓ Confidence: {result['confidence']:.2%}")

                print("\nTop 3 Predictions:")
                for i, (cat, prob) in enumerate(result['top_3_predictions'], 1):
                    print(f"  {i}. {cat}: {prob:.2%}")

            print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n\nBeendet.")
            break
        except Exception as e:
            print(f"\n❌ Fehler: {e}\n")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Rechnungs-Klassifizierung mit trainiertem Model"
    )

    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch', 'submission'],
        default='interactive',
        help='Modus: interactive (einzelne Eingaben), batch (CSV), submission (für Competition)'
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Input CSV-Datei für batch/submission Modus'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output CSV-Datei für batch/submission Modus'
    )

    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Verzeichnis mit gespeicherten Models'
    )

    args = parser.parse_args()

    # Classifier laden
    classifier = InvoiceClassifier(model_dir=args.model_dir)

    # Modus ausführen
    if args.mode == 'interactive':
        interactive_mode(classifier)

    elif args.mode == 'batch':
        if not args.input:
            print("❌ Fehler: --input erforderlich für batch Modus")
            return

        results = classifier.predict_batch(args.input)
        results.to_csv(args.output, index=False)
        print(f"✓ Ergebnisse gespeichert: {args.output}")

        # Übersicht
        print("\nKategorieverteilung:")
        print(results['Predicted_Category'].value_counts().head(10))

    elif args.mode == 'submission':
        if not args.input:
            print("❌ Fehler: --input erforderlich für submission Modus")
            return

        classifier.predict_and_save_submission(args.input, args.output)


if __name__ == "__main__":
    main()
