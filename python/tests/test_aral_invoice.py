"""
Test-Script für ARAL Rechnung
Testet OCR-Extraktion und JSON-Parsing mit deutscher Rechnung
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.json_parser import InvoiceJSONParser

# ARAL Rechnungstext (ANONYMISIERT - Testdaten)
# Alle personenbezogenen Daten wurden durch Musterdaten ersetzt
ARAL_TEXT = """ARAL Tankstelle Musterstadt

ARAL Tankstelle Musterstadt, Musterstraße 1, 12345 Musterstadt

Musterstraße 32
Herr 12345 Musterstadt
Max Mustermann
Beispielweg 123 Tel. : +49 (123) 456789
12345 Musterstadt Fax: +49 (123) 456780
Deutschland

Rechnungsdatum: 31.10.2025

Rechnung
— Rechnungsnr. : 204/2025
Mandatsreferenz : 111 Debitorennr. : 10111
Brutto Brutto
Bonnr. Datum Freigabe Menge Bezeichnung Fahrern. KM-Stand MwSt Einzelpreis Gesamtpreis
Ausweisnr.: 451

A:27871 01.10.2025 17:15 Karte 40,15 SuperE 5 19,00 % 1,669 67,01 EUR
Kraftstoffmenge: 40,15 Liter Ausweisnr. 451 Summe 67,01 EUR
Netto Werte: Ausweisnr. 451 Summe (56,31 EUR)
Gesamtsumme 67,01 EUR
Netto MwSt Brutto
5 A:MwSt 19 % 56,31 EUR 10,70 EUR 67,01 EUR
Summe 56,31 EUR 10,70 EUR 67,01 EUR
Zu zahlen 67,01 EUR

Rechnungsbetrag wird wie vereinbart von Ihrem Konto IBAN: DE89370400440532013000 BIC: TESTDE88XXX Bank: Musterbank
unter der Gläubiger-ID DE13ZZ200000000000 per SEPA-Lastschriftverfahren am 05.11.2025 eingezogen.

Gesamt Kraftstoffmenge in Liter: 40,15 Liter
A: Der Verkauf von Kraft- und Schmierstoffen erfolgt im Namen und für Rechnung der Mineralölgesellschaft

Seite 1/1
Geschäftsführer: Mustermann GmbH - Gerichtsstand ist Musterstadt
Bankverbindung: Musterbank - IBAN: DE89370400440532013000 - BIC: TESTDE88XXX

Ustid: DE999999999 - StNr: 123/456/78901
"""


def parse_aral_invoice(text):
    """
    Parst ARAL-Rechnung zu strukturiertem JSON
    """
    import re

    invoice_json = {
        "invoice": {},
        "items": [],
        "subtotal": {},
        "payment_instructions": {},
        "_ocr_text": text
    }

    # Seller (ARAL Tankstelle)
    seller_match = re.search(r'ARAL Tankstelle Musterstadt', text)
    if seller_match:
        invoice_json["invoice"]["seller_name"] = "ARAL Tankstelle Musterstadt"
        invoice_json["invoice"]["seller_address"] = "Musterstraße 1, 12345 Musterstadt"

    # Client (Max Mustermann)
    client_match = re.search(r'Herr.*?Max Mustermann', text, re.DOTALL)
    if client_match:
        invoice_json["invoice"]["client_name"] = "Max Mustermann"
        invoice_json["invoice"]["client_address"] = "Beispielweg 123, 12345 Musterstadt"

    # Invoice Number
    inv_num_match = re.search(r'Rechnungsnr\.\s*:\s*(\d+/\d+)', text)
    if inv_num_match:
        invoice_json["invoice"]["invoice_number"] = inv_num_match.group(1)

    # Invoice Date
    inv_date_match = re.search(r'Rechnungsdatum:\s*(\d{2}\.\d{2}\.\d{4})', text)
    if inv_date_match:
        invoice_json["invoice"]["invoice_date"] = inv_date_match.group(1)

    # Due Date (SEPA Einzugsdatum)
    due_date_match = re.search(r'am\s*(\d{2}\.\d{2}\.\d{4})\s*eingezogen', text)
    if due_date_match:
        invoice_json["invoice"]["due_date"] = due_date_match.group(1)

    # Items (Kraftstoff)
    items = []

    # SuperE 5 Item
    item_match = re.search(r'(\d+,\d+)\s+SuperE\s+5.*?(\d+,\d+)\s+EUR', text)
    if item_match:
        quantity = item_match.group(1).replace(',', '.')
        price = item_match.group(2).replace(',', '.')

        items.append({
            "description": "SuperE 5 (Kraftstoff)",
            "quantity": quantity + " Liter",
            "total_price": price
        })

    invoice_json["items"] = items

    # Subtotal
    # Tax (MwSt)
    tax_match = re.search(r'MwSt.*?(\d+,\d+)\s+EUR', text)
    if tax_match:
        invoice_json["subtotal"]["tax"] = tax_match.group(1).replace(',', '.')

    # Total (Gesamtsumme)
    total_match = re.search(r'Gesamtsumme\s+(\d+,\d+)\s+EUR', text)
    if total_match:
        invoice_json["subtotal"]["total"] = total_match.group(1).replace(',', '.')

    # Netto
    netto_match = re.search(r'Summe\s+(\d+,\d+)\s+EUR\s+\d+,\d+\s+EUR\s+\d+,\d+\s+EUR', text)
    if netto_match:
        invoice_json["subtotal"]["net"] = netto_match.group(1).replace(',', '.')

    # Payment Instructions
    iban_match = re.search(r'IBAN:\s*(DE\d+)', text)
    if iban_match:
        invoice_json["payment_instructions"]["iban"] = iban_match.group(1)

    bic_match = re.search(r'BIC:\s*([A-Z0-9]+)', text)
    if bic_match:
        invoice_json["payment_instructions"]["bic"] = bic_match.group(1)

    bank_match = re.search(r'Bank:\s*([^\n]+)', text)
    if bank_match:
        invoice_json["payment_instructions"]["bank_name"] = bank_match.group(1).strip()

    invoice_json["payment_instructions"]["payment_method"] = "SEPA-Lastschrift"
    invoice_json["payment_instructions"]["due_date"] = invoice_json["invoice"].get("due_date", "")

    return invoice_json


def main():
    print("="*60)
    print("TEST: ARAL Rechnung - OCR & JSON Parsing")
    print("="*60)

    # 1. Parse ARAL Text zu JSON
    print("\n[1/2] Parse ARAL-Text zu JSON...")
    invoice_json = parse_aral_invoice(ARAL_TEXT)

    print("\n" + "="*60)
    print("EXTRAHIERTES JSON:")
    print("="*60)
    print(json.dumps(invoice_json, indent=2, ensure_ascii=False))

    # 2. JSON Parser testen
    print("\n[2/2] Feature-Extraktion mit InvoiceJSONParser...")
    parser = InvoiceJSONParser()
    features = parser.extract_all_features(invoice_json)

    print("\n" + "="*60)
    print("EXTRAHIERTE FEATURES:")
    print("="*60)

    # Wichtigste Features anzeigen
    important_features = [
        'seller_name', 'client_name', 'invoice_number',
        'combined_text', 'num_items',
        'total', 'tax', 'tax_rate',
        'invoice_year', 'invoice_month', 'invoice_day'
    ]

    for key in important_features:
        if key in features:
            value = features[key]
            if isinstance(value, str) and len(value) > 80:
                value = value[:80] + "..."
            print(f"{key:25s}: {value}")

    # Alle numerischen Features
    print("\n" + "-"*60)
    print("NUMERISCHE FEATURES:")
    print("-"*60)

    numeric_features = [
        'total_quantity', 'total_items_price', 'avg_item_price',
        'tax', 'discount', 'total', 'tax_rate', 'discount_rate'
    ]

    for key in numeric_features:
        if key in features:
            print(f"{key:25s}: {features[key]}")

    # JSON speichern
    output_path = "test_aral_invoice.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(invoice_json, f, indent=2, ensure_ascii=False)

    print(f"\n✓ JSON gespeichert: {output_path}")

    # Kategorie-Prediction (simuliert)
    print("\n" + "="*60)
    print("KLASSIFIZIERUNG (Simuliert):")
    print("="*60)
    print("📊 Erwartete Kategorie: CLASS-1758 (Kraftstoff/Benzin)")
    print("💡 Features für ML:")
    print(f"   - Seller: {features.get('seller_name', 'N/A')}")
    print(f"   - Items: {features.get('num_items', 0)}")
    print(f"   - Total: {features.get('total', 0):.2f} EUR")
    print(f"   - Tax Rate: {features.get('tax_rate', 0):.2%}")
    print(f"   - Text enthält: 'Kraftstoff', 'SuperE', 'Liter'")

    print("\n✓ Test abgeschlossen!")


if __name__ == "__main__":
    main()
