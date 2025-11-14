"""
OCR Extraction Module für Rechnungsbilder
Extrahiert Text und erstellt strukturiertes JSON
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False


class InvoiceOCRExtractor:
    """
    OCR Extractor für Rechnungsbilder
    """

    def __init__(self, ocr_engine='easyocr', languages=['en', 'de'], use_gpu=True):
        """
        Args:
            ocr_engine: 'easyocr' oder 'tesseract'
            languages: Liste der Sprachen für OCR (default: Englisch + Deutsch)
            use_gpu: GPU verwenden (True für M3/CUDA)
        """
        self.ocr_engine = ocr_engine.lower()
        self.languages = languages
        self.use_gpu = use_gpu

        # OCR Engine initialisieren
        if self.ocr_engine == 'easyocr':
            if not EASYOCR_AVAILABLE:
                raise ImportError("EasyOCR nicht installiert. Installiere mit: pip install easyocr")
            print(f"Initialisiere EasyOCR mit Sprachen: {languages}")
            print(f"GPU: {use_gpu}")
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
        elif self.ocr_engine == 'tesseract':
            if not PYTESSERACT_AVAILABLE:
                raise ImportError("pytesseract nicht installiert. Installiere mit: pip install pytesseract")
            print(f"Verwende Tesseract OCR")
            # Tesseract Config für Deutsch
            self.tesseract_config = '--oem 3 --psm 6 -l deu+eng'
        else:
            raise ValueError(f"Unbekannte OCR Engine: {ocr_engine}")

    def preprocess_image(self, image_path):
        """
        Vorverarbeitung des Bildes für bessere OCR-Ergebnisse

        Args:
            image_path: Pfad zum Bild

        Returns:
            Vorverarbeitetes Bild (numpy array)
        """
        # Bild laden
        img = cv2.imread(str(image_path))

        # Zu Graustufen konvertieren
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Kontrast verbessern (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Rauschen reduzieren
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)

        # Binarisierung (Otsu's method)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def extract_text(self, image_path):
        """
        Extrahiert Text aus Bild

        Args:
            image_path: Pfad zum Bild

        Returns:
            Extrahierter Text als String
        """
        # Bild vorverarbeiten
        processed_img = self.preprocess_image(image_path)

        if self.ocr_engine == 'easyocr':
            # EasyOCR
            results = self.reader.readtext(processed_img, detail=0, paragraph=True)
            text = "\n".join(results)
        else:
            # Tesseract
            text = pytesseract.image_to_string(processed_img)

        return text

    def extract_invoice_number(self, text):
        """Extrahiert Rechnungsnummer"""
        patterns = [
            r'invoice\s*(?:no|number|#)[:\s]*(\d+)',
            r'invoice[:\s]*(\d+)',
            r'inv\s*(?:no|#)[:\s]*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def extract_date(self, text, field_name='invoice'):
        """Extrahiert Datum (verschiedene Formate)"""
        patterns = [
            r'(?:date|' + field_name + r')[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return ""

    def extract_seller_client_info(self, text):
        """Extrahiert Seller und Client Informationen"""
        seller_name = ""
        seller_address = ""
        client_name = ""
        client_address = ""

        # Seller
        seller_match = re.search(r'seller[:\s]*([^\n]+)', text, re.IGNORECASE)
        if seller_match:
            seller_name = seller_match.group(1).strip()

        # Client
        client_match = re.search(r'client[:\s]*([^\n]+)', text, re.IGNORECASE)
        if client_match:
            client_name = client_match.group(1).strip()

        return seller_name, seller_address, client_name, client_address

    def extract_items(self, text):
        """
        Extrahiert Line Items aus Rechnung
        Sehr vereinfacht - kann je nach Rechnungsformat angepasst werden
        """
        items = []

        # Suche nach Tabellen-Zeilen mit Preis
        lines = text.split('\n')

        for line in lines:
            # Preis-Pattern (Betrag mit 2 Dezimalstellen)
            price_match = re.search(r'(\d+[.,]\d{2})', line)
            if price_match:
                # Versuche Menge zu finden
                qty_match = re.search(r'(\d+[.,]?\d*)\s*(?:each|pcs?|x)', line, re.IGNORECASE)
                quantity = qty_match.group(1) if qty_match else "1.00"

                # Rest als Beschreibung
                description = line.strip()

                items.append({
                    "description": description,
                    "quantity": quantity,
                    "total_price": price_match.group(1)
                })

        return items

    def extract_totals(self, text):
        """Extrahiert Summen (Tax, Discount, Total)"""
        tax = ""
        discount = ""
        total = ""

        # Tax
        tax_match = re.search(r'(?:tax|vat)[:\s]*\$?\s*(\d+[.,]\d{2})', text, re.IGNORECASE)
        if tax_match:
            tax = tax_match.group(1)

        # Discount
        discount_match = re.search(r'discount[:\s]*\$?\s*(\d+[.,]\d{2})', text, re.IGNORECASE)
        if discount_match:
            discount = discount_match.group(1)

        # Total
        total_match = re.search(r'(?:total|gross)[:\s]*\$?\s*(\d+[.,]\d{2})', text, re.IGNORECASE)
        if total_match:
            total = total_match.group(1)

        return tax, discount, total

    def extract_to_json(self, image_path):
        """
        Haupt-Methode: Extrahiert alle Daten und erstellt JSON

        Args:
            image_path: Pfad zum Rechnungsbild

        Returns:
            dict mit strukturierten Rechnungsdaten
        """
        print(f"Verarbeite Bild: {image_path}")

        # OCR durchführen
        text = self.extract_text(image_path)

        # Strukturierte Daten extrahieren
        invoice_number = self.extract_invoice_number(text)
        invoice_date = self.extract_date(text, 'invoice')
        due_date = self.extract_date(text, 'due')

        seller_name, seller_address, client_name, client_address = \
            self.extract_seller_client_info(text)

        items = self.extract_items(text)
        tax, discount, total = self.extract_totals(text)

        # JSON-Struktur erstellen
        invoice_json = {
            "invoice": {
                "client_name": client_name,
                "client_address": client_address,
                "seller_name": seller_name,
                "seller_address": seller_address,
                "invoice_number": invoice_number,
                "invoice_date": invoice_date,
                "due_date": due_date
            },
            "items": items,
            "subtotal": {
                "tax": tax,
                "discount": discount,
                "total": total
            },
            "payment_instructions": {
                "due_date": due_date,
                "bank_name": "",
                "account_number": "",
                "payment_method": ""
            }
        }

        # Raw OCR Text hinzufügen
        invoice_json["_ocr_text"] = text

        return invoice_json

    def save_json(self, json_data, output_path):
        """Speichert JSON zu Datei"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✓ JSON gespeichert: {output_path}")


def batch_process_invoices(image_dir, output_dir, ocr_engine='easyocr'):
    """
    Verarbeitet mehrere Rechnungsbilder in einem Batch

    Args:
        image_dir: Verzeichnis mit Rechnungsbildern
        output_dir: Verzeichnis für JSON-Outputs
        ocr_engine: 'easyocr' oder 'tesseract'
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Alle Bilder finden
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Gefunden: {len(image_files)} Bilder")

    # OCR Extractor
    extractor = InvoiceOCRExtractor(ocr_engine=ocr_engine)

    # Batch-Verarbeitung
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")

        try:
            # OCR → JSON
            json_data = extractor.extract_to_json(img_path)

            # JSON speichern
            json_filename = img_path.stem + ".json"
            output_path = output_dir / json_filename
            extractor.save_json(json_data, output_path)

        except Exception as e:
            print(f"❌ Fehler bei {img_path.name}: {e}")
            continue

    print(f"\n✓ Batch-Verarbeitung abgeschlossen!")
    print(f"  {len(image_files)} Bilder verarbeitet")
    print(f"  Output: {output_dir}")


def main():
    """
    Beispiel-Verwendung
    """
    import argparse

    parser = argparse.ArgumentParser(description="OCR für Rechnungsbilder")
    parser.add_argument('--image', type=str, help='Einzelnes Bild verarbeiten')
    parser.add_argument('--batch', type=str, help='Verzeichnis mit Bildern (Batch)')
    parser.add_argument('--output', type=str, default='output_json', help='Output-Verzeichnis')
    parser.add_argument('--engine', type=str, default='easyocr', choices=['easyocr', 'tesseract'])

    args = parser.parse_args()

    if args.batch:
        # Batch-Verarbeitung
        batch_process_invoices(args.batch, args.output, args.engine)

    elif args.image:
        # Einzelnes Bild
        extractor = InvoiceOCRExtractor(ocr_engine=args.engine)
        json_data = extractor.extract_to_json(args.image)

        # JSON ausgeben
        print("\n" + "="*60)
        print("EXTRAHIERTES JSON:")
        print("="*60)
        print(json.dumps(json_data, indent=2, ensure_ascii=False))

        # Optional speichern
        if args.output:
            output_path = Path(args.output) / (Path(args.image).stem + ".json")
            extractor.save_json(json_data, output_path)

    else:
        print("❌ Bitte --image oder --batch angeben")


if __name__ == "__main__":
    main()
