"""
End-to-End Inference für Rechnungsklassifizierung
Workflow: Bild → OCR → JSON → PyTorch Model → Kategorie
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ocr.extractor import InvoiceOCRExtractor
from src.data.json_parser import InvoiceJSONParser
from src.models.invoice_model import create_model


class InvoicePredictor:
    """
    End-to-End Predictor für Rechnungen
    """

    def __init__(
        self,
        model_path,
        model_config,
        device='auto',
        ocr_engine='easyocr',
        use_gpu_ocr=True
    ):
        """
        Args:
            model_path: Pfad zum gespeicherten Model (.pth)
            model_config: dict mit Model-Konfiguration
            device: 'cuda', 'mps', 'cpu', or 'auto' (auto-detect)
            ocr_engine: 'easyocr' oder 'tesseract'
            use_gpu_ocr: GPU für OCR verwenden
        """
        # Auto-detect device (M3 GPU Support)
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🚀 Verwende CUDA GPU")
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
                print("🚀 Verwende Apple M3 GPU (MPS)")
            else:
                self.device = torch.device('cpu')
                print("⚠️  Verwende CPU")
        else:
            self.device = torch.device(device)

        self.model_config = model_config
        print(f"Device: {self.device}")

        # OCR Extractor (mit GPU falls M3/CUDA)
        print(f"Initialisiere OCR Engine: {ocr_engine}")
        print(f"OCR GPU: {use_gpu_ocr}")
        self.ocr_extractor = InvoiceOCRExtractor(
            ocr_engine=ocr_engine,
            languages=['en', 'de'],  # Deutsch + Englisch
            use_gpu=use_gpu_ocr
        )

        # JSON Parser
        self.json_parser = InvoiceJSONParser()

        # Model laden
        print(f"Lade Model: {model_path}")
        self.model = self._load_model(model_path, model_config)
        self.model.eval()

        # Tokenizer (falls Text verwendet wird)
        self.tokenizer = None
        if model_config.get('use_text', False):
            text_model = model_config.get('text_model', 'distilbert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(text_model)
            self.max_length = 128

        # Image Transform
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Label Encoder (muss beim Training gespeichert worden sein)
        self.label_classes = model_config.get('label_classes', None)

        print("✓ Predictor initialisiert!")

    def _load_model(self, model_path, config):
        """Lädt gespeichertes Model"""
        # Model erstellen
        model = create_model(
            num_classes=config['num_classes'],
            model_type=config.get('model_type', 'multimodal'),
            vision_backbone=config.get('vision_backbone', 'resnet50'),
            text_model=config.get('text_model', 'distilbert-base-uncased'),
            numeric_input_dim=config.get('numeric_input_dim', 15),
            use_vision=config.get('use_vision', True),
            use_text=config.get('use_text', True),
            use_numeric=config.get('use_numeric', True)
        )

        # Weights laden
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        return model

    def predict_from_image(self, image_path, return_json=False):
        """
        Klassifiziert Rechnung direkt vom Bild

        Args:
            image_path: Pfad zum Rechnungsbild
            return_json: Ob extrahiertes JSON zurückgegeben werden soll

        Returns:
            dict mit prediction, confidence, (optional: json_data)
        """
        print(f"\nVerarbeite: {image_path}")

        # Schritt 1: OCR → JSON
        print("  [1/4] OCR Extraktion...")
        json_data = self.ocr_extractor.extract_to_json(image_path)

        # Schritt 2: JSON → Features
        print("  [2/4] Feature-Extraktion...")
        features = self.json_parser.extract_all_features(json_data)

        # Schritt 3: Bild laden
        print("  [3/4] Bild-Preprocessing...")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # Text Features
        text = features.get('combined_text', '')

        # Numerische Features
        numeric_feature_names = [
            'total_quantity', 'total_items_price', 'avg_item_price', 'num_items',
            'tax', 'discount', 'total', 'tax_rate', 'discount_rate',
            'invoice_year', 'invoice_month', 'invoice_day', 'invoice_weekday',
            'invoice_quarter', 'days_until_due'
        ]

        numeric_values = []
        for name in numeric_feature_names:
            value = features.get(name, 0.0)
            if value is None or np.isnan(value):
                value = 0.0
            numeric_values.append(float(value))

        numeric_tensor = torch.tensor([numeric_values], dtype=torch.float32).to(self.device)

        # Schritt 4: Model Inference
        print("  [4/4] Model Inference...")

        with torch.no_grad():
            # Text tokenisieren (falls verwendet)
            input_ids = None
            attention_mask = None

            if self.tokenizer is not None:
                encoded = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

            # Model Forward
            if self.model_config.get('model_type') == 'vision_only':
                logits = self.model(image_tensor)
            else:
                logits = self.model(
                    image=image_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numeric_features=numeric_tensor
                )

            # Prediction
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

            pred_idx = pred_idx.item()
            confidence = confidence.item()

        # Klasse dekodieren
        if self.label_classes is not None:
            predicted_class = self.label_classes[pred_idx]
        else:
            predicted_class = f"CLASS-{pred_idx}"

        # Top-3 Predictions
        top3_probs, top3_indices = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
        top3_predictions = []

        for i in range(top3_probs.shape[1]):
            idx = top3_indices[0, i].item()
            prob = top3_probs[0, i].item()

            if self.label_classes is not None:
                class_name = self.label_classes[idx]
            else:
                class_name = f"CLASS-{idx}"

            top3_predictions.append((class_name, prob))

        # Result
        result = {
            'predicted_category': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top3_predictions
        }

        if return_json:
            result['json_data'] = json_data

        return result

    def predict_from_json(self, json_data, image_path=None):
        """
        Klassifiziert Rechnung direkt von JSON-Daten
        (wenn OCR bereits durchgeführt wurde)

        Args:
            json_data: JSON-Daten (dict oder String)
            image_path: Optional - Pfad zum Bild (falls Vision verwendet wird)

        Returns:
            dict mit prediction, confidence
        """
        # Features extrahieren
        features = self.json_parser.extract_all_features(json_data)

        # Text
        text = features.get('combined_text', '')

        # Numeric Features
        numeric_feature_names = [
            'total_quantity', 'total_items_price', 'avg_item_price', 'num_items',
            'tax', 'discount', 'total', 'tax_rate', 'discount_rate',
            'invoice_year', 'invoice_month', 'invoice_day', 'invoice_weekday',
            'invoice_quarter', 'days_until_due'
        ]

        numeric_values = []
        for name in numeric_feature_names:
            value = features.get(name, 0.0)
            if value is None or np.isnan(value):
                value = 0.0
            numeric_values.append(float(value))

        numeric_tensor = torch.tensor([numeric_values], dtype=torch.float32).to(self.device)

        # Bild (falls vorhanden)
        image_tensor = None
        if image_path and self.model_config.get('use_vision', True):
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.image_transform(image).unsqueeze(0).to(self.device)

        # Model Inference
        with torch.no_grad():
            # Text tokenisieren
            input_ids = None
            attention_mask = None

            if self.tokenizer is not None:
                encoded = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

            # Forward
            logits = self.model(
                image=image_tensor,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_tensor
            )

            # Prediction
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

            pred_idx = pred_idx.item()
            confidence = confidence.item()

        # Klasse
        if self.label_classes is not None:
            predicted_class = self.label_classes[pred_idx]
        else:
            predicted_class = f"CLASS-{pred_idx}"

        return {
            'predicted_category': predicted_class,
            'confidence': confidence
        }


def main():
    """
    CLI für Inference
    """
    parser = argparse.ArgumentParser(description="Rechnungsklassifizierung mit PyTorch")

    parser.add_argument('--image', type=str, help='Pfad zum Rechnungsbild')
    parser.add_argument('--model', type=str, required=True, help='Pfad zum gespeicherten Model (.pth)')
    parser.add_argument('--config', type=str, required=True, help='Pfad zur Model Config (.json)')
    parser.add_argument('--ocr-engine', type=str, default='easyocr', choices=['easyocr', 'tesseract'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-json', action='store_true', help='Speichere extrahiertes JSON')

    args = parser.parse_args()

    # Config laden
    with open(args.config, 'r') as f:
        model_config = json.load(f)

    # Predictor
    predictor = InvoicePredictor(
        model_path=args.model,
        model_config=model_config,
        device=args.device,
        ocr_engine=args.ocr_engine
    )

    # Prediction
    if args.image:
        result = predictor.predict_from_image(args.image, return_json=args.save_json)

        # Ausgabe
        print("\n" + "="*60)
        print("ERGEBNIS:")
        print("="*60)
        print(f"📊 Kategorie: {result['predicted_category']}")
        print(f"✓ Confidence: {result['confidence']:.2%}")

        print("\nTop 3 Predictions:")
        for i, (cat, prob) in enumerate(result['top_3_predictions'], 1):
            print(f"  {i}. {cat}: {prob:.2%}")

        # JSON speichern (optional)
        if args.save_json and 'json_data' in result:
            json_path = Path(args.image).stem + "_extracted.json"
            with open(json_path, 'w') as f:
                json.dump(result['json_data'], f, indent=2)
            print(f"\n✓ JSON gespeichert: {json_path}")

    else:
        print("❌ Bitte --image angeben")


if __name__ == "__main__":
    main()
