"""
JSON Parser für extrahierte Rechnungsdaten
Konvertiert JSON zu strukturierten Features für ML
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime


class InvoiceJSONParser:
    """
    Parser für Rechnungs-JSON Daten
    """

    def __init__(self):
        """Initialisierung"""
        pass

    def parse_json(self, json_data):
        """
        Parst JSON-Daten (String oder Dict)

        Args:
            json_data: JSON als String oder dict

        Returns:
            Geparste dict-Struktur
        """
        if isinstance(json_data, str):
            try:
                return json.loads(json_data)
            except json.JSONDecodeError as e:
                print(f"JSON Parse Error: {e}")
                return {}
        return json_data

    def extract_text_features(self, invoice_json):
        """
        Extrahiert Text-Features für NLP/ML

        Args:
            invoice_json: Geparste JSON-Daten

        Returns:
            dict mit Text-Features
        """
        features = {}

        # Invoice Header Text
        invoice_info = invoice_json.get('invoice', {})
        features['seller_name'] = invoice_info.get('seller_name', '')
        features['client_name'] = invoice_info.get('client_name', '')
        features['invoice_number'] = invoice_info.get('invoice_number', '')

        # Items Beschreibungen zusammenführen
        items = invoice_json.get('items', [])
        descriptions = []
        for item in items:
            desc = item.get('description', '')
            if desc:
                descriptions.append(desc)

        features['item_descriptions'] = ' '.join(descriptions)
        features['num_items'] = len(items)

        # Kombinierter Text für TF-IDF/BERT
        combined_text = f"{features['seller_name']} {features['client_name']} {features['item_descriptions']}"
        features['combined_text'] = combined_text

        # Raw OCR Text (falls vorhanden)
        features['ocr_text'] = invoice_json.get('_ocr_text', '')

        return features

    def extract_numeric_features(self, invoice_json):
        """
        Extrahiert numerische Features

        Args:
            invoice_json: Geparste JSON-Daten

        Returns:
            dict mit numerischen Features
        """
        features = {}

        # Items
        items = invoice_json.get('items', [])

        total_quantity = 0.0
        total_price = 0.0
        avg_price = 0.0

        for item in items:
            # Quantity
            qty_str = item.get('quantity', '0')
            qty = self._parse_number(qty_str)
            total_quantity += qty

            # Price
            price_str = item.get('total_price', '0')
            price = self._parse_number(price_str)
            total_price += price

        if len(items) > 0:
            avg_price = total_price / len(items)

        features['total_quantity'] = total_quantity
        features['total_items_price'] = total_price
        features['avg_item_price'] = avg_price
        features['num_items'] = len(items)

        # Subtotal
        subtotal = invoice_json.get('subtotal', {})
        features['tax'] = self._parse_number(subtotal.get('tax', '0'))
        features['discount'] = self._parse_number(subtotal.get('discount', '0'))
        features['total'] = self._parse_number(subtotal.get('total', '0'))

        # Berechnete Features
        features['tax_rate'] = features['tax'] / features['total'] if features['total'] > 0 else 0
        features['discount_rate'] = features['discount'] / features['total'] if features['total'] > 0 else 0

        return features

    def extract_date_features(self, invoice_json):
        """
        Extrahiert Datum-Features

        Args:
            invoice_json: Geparste JSON-Daten

        Returns:
            dict mit Datum-Features
        """
        features = {}

        invoice_info = invoice_json.get('invoice', {})

        # Invoice Date
        invoice_date_str = invoice_info.get('invoice_date', '')
        invoice_date = self._parse_date(invoice_date_str)

        if invoice_date:
            features['invoice_year'] = invoice_date.year
            features['invoice_month'] = invoice_date.month
            features['invoice_day'] = invoice_date.day
            features['invoice_weekday'] = invoice_date.weekday()
            features['invoice_quarter'] = (invoice_date.month - 1) // 3 + 1
        else:
            features['invoice_year'] = 0
            features['invoice_month'] = 0
            features['invoice_day'] = 0
            features['invoice_weekday'] = 0
            features['invoice_quarter'] = 0

        # Due Date (ähnlich)
        due_date_str = invoice_info.get('due_date', '')
        due_date = self._parse_date(due_date_str)

        if due_date and invoice_date:
            # Tage bis Fälligkeit
            features['days_until_due'] = (due_date - invoice_date).days
        else:
            features['days_until_due'] = 0

        return features

    def extract_all_features(self, invoice_json):
        """
        Extrahiert alle Features

        Args:
            invoice_json: Geparste JSON-Daten (String oder dict)

        Returns:
            dict mit allen Features
        """
        # Parse JSON falls String
        parsed_json = self.parse_json(invoice_json)

        # Features sammeln
        all_features = {}

        # Text Features
        text_features = self.extract_text_features(parsed_json)
        all_features.update(text_features)

        # Numerische Features
        numeric_features = self.extract_numeric_features(parsed_json)
        all_features.update(numeric_features)

        # Datum Features
        date_features = self.extract_date_features(parsed_json)
        all_features.update(date_features)

        return all_features

    def create_feature_dataframe(self, json_list):
        """
        Erstellt DataFrame aus Liste von JSON-Daten

        Args:
            json_list: Liste von JSON-Daten (String oder dict)

        Returns:
            pandas DataFrame mit Features
        """
        feature_list = []

        for json_data in json_list:
            features = self.extract_all_features(json_data)
            feature_list.append(features)

        df = pd.DataFrame(feature_list)
        return df

    # Hilfsfunktionen
    def _parse_number(self, number_str):
        """
        Parst Zahlen-String zu float
        Unterstützt: "123.45", "123,45", "$123.45", etc.
        """
        if not number_str:
            return 0.0

        # String bereinigen
        cleaned = str(number_str).strip()
        # Entferne $, €, etc.
        cleaned = re.sub(r'[^\d.,\-]', '', cleaned)
        # Komma zu Punkt
        cleaned = cleaned.replace(',', '.')

        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _parse_date(self, date_str):
        """
        Parst Datum-String zu datetime
        Unterstützt verschiedene Formate
        """
        if not date_str:
            return None

        # Verschiedene Formate probieren
        formats = [
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d',
            '%m-%d-%Y',
            '%d.%m.%Y',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        return None

    def load_from_csv(self, csv_path, json_column='Json Data'):
        """
        Lädt JSON-Daten aus CSV (wie batch1_1.csv)

        Args:
            csv_path: Pfad zur CSV-Datei
            json_column: Name der Spalte mit JSON-Daten

        Returns:
            pandas DataFrame mit extrahierten Features
        """
        print(f"Lade CSV: {csv_path}")
        df_csv = pd.read_csv(csv_path)

        print(f"Zeilen: {len(df_csv)}")

        # JSON parsen und Features extrahieren
        json_data_list = df_csv[json_column].tolist()
        features_df = self.create_feature_dataframe(json_data_list)

        # File Name hinzufügen
        if 'File Name' in df_csv.columns:
            features_df['file_name'] = df_csv['File Name'].values

        print(f"Features extrahiert: {features_df.shape[1]} Features")

        return features_df


def main():
    """
    Beispiel-Verwendung
    """
    import argparse

    parser = argparse.ArgumentParser(description="JSON Parser für Rechnungen")
    parser.add_argument('--json', type=str, help='Einzelne JSON-Datei parsen')
    parser.add_argument('--csv', type=str, help='CSV mit JSON-Daten parsen')
    parser.add_argument('--output', type=str, help='Output CSV für Features')

    args = parser.parse_args()

    parser_obj = InvoiceJSONParser()

    if args.json:
        # Einzelne JSON-Datei
        with open(args.json, 'r') as f:
            json_data = json.load(f)

        features = parser_obj.extract_all_features(json_data)

        print("\n" + "="*60)
        print("EXTRAHIERTE FEATURES:")
        print("="*60)
        for key, value in features.items():
            print(f"{key:25s}: {value}")

    elif args.csv:
        # CSV mit JSON-Daten
        features_df = parser_obj.load_from_csv(args.csv)

        print("\n" + "="*60)
        print("FEATURE DATAFRAME:")
        print("="*60)
        print(features_df.head())
        print(f"\nShape: {features_df.shape}")
        print(f"\nSpalten: {list(features_df.columns)}")

        # Optional speichern
        if args.output:
            features_df.to_csv(args.output, index=False)
            print(f"\n✓ Features gespeichert: {args.output}")

    else:
        print("❌ Bitte --json oder --csv angeben")


if __name__ == "__main__":
    main()
