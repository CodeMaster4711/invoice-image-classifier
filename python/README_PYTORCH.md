# PyTorch Rechnungsklassifizierung - Multimodal (Bild + JSON)

End-to-End Pipeline: **Rechnungsbild → OCR → JSON → PyTorch Model → Kategorie**

## 🚀 Features

- **OCR Extraktion**: EasyOCR/Tesseract für Text-Extraktion aus Bildern
- **JSON Parsing**: Strukturierte Datenextraktion (Invoice-Header, Items, Totals)
- **Multimodal Model**:
  - **Vision**: ResNet/EfficientNet für Bildverarbeitung
  - **Text**: BERT/DistilBERT für JSON-Text
  - **Numeric**: MLP für numerische Features (Beträge, Datum, etc.)
- **PyTorch Training**: Vollständige Training-Pipeline mit DataLoader
- **End-to-End Inference**: Bild → Klassifikation in einem Schritt

## 📦 Installation

```bash
# Dependencies installieren
pip install torch torchvision transformers
pip install easyocr  # oder: pytesseract
pip install opencv-python pillow pandas scikit-learn tqdm
```

## 📁 Projektstruktur

```
python/
├── ocr_extractor.py              # OCR: Bild → JSON
├── json_parser.py                # JSON → Features
├── invoice_dataset.py            # PyTorch Dataset
├── invoice_model.py              # Multimodal Model
├── train_pytorch.py              # Training Pipeline
├── predict_invoice_pytorch.py    # Inference Pipeline
└── pytorch_models/               # Gespeicherte Models
```

## 🔧 Workflow

### 1. OCR: Bild → JSON

Extrahiere Text aus Rechnungsbildern und erstelle strukturiertes JSON:

```bash
# Einzelnes Bild
python ocr_extractor.py --image batch1-0001.jpg --output json_output/

# Batch-Verarbeitung
python ocr_extractor.py --batch ../Data/batch_1/batch_1/batch1_1/ --output json_output/
```

**Output JSON Format:**
```json
{
  "invoice": {
    "client_name": "Becker Ltd",
    "seller_name": "Andrews, Kirby and Valdez",
    "invoice_number": "51109338",
    "invoice_date": "04/13/2013"
  },
  "items": [
    {
      "description": "CLEARANCE! Fast Dell Desktop...",
      "quantity": "3.00",
      "total_price": "627.00"
    }
  ],
  "subtotal": {
    "tax": "564.02",
    "total": "6204.19"
  }
}
```

### 2. JSON → Features

Parse JSON zu ML-Features:

```bash
# Aus CSV mit JSON-Daten
python json_parser.py --csv ../Data/batch_1/batch_1/batch1_1.csv --output features.csv
```

**Extrahierte Features:**
- **Text**: seller_name, client_name, item_descriptions, combined_text
- **Numeric**: total_quantity, total_price, tax, discount, tax_rate
- **Date**: invoice_year, invoice_month, invoice_day, invoice_quarter

### 3. Training

Trainiere PyTorch Multimodal Model:

```bash
python train_pytorch.py \
  --train-csv ../Data/batch_1/batch_1/batch1_1.csv \
  --train-images ../Data/batch_1/batch_1/batch1_1/ \
  --labels-csv ../Data/Dataset/Train.csv \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-4 \
  --model-type multimodal \
  --vision-backbone resnet50 \
  --text-model distilbert-base-uncased \
  --save-dir pytorch_models/
```

**Model Typen:**
- `multimodal`: Vision + Text + Numeric (Best Performance)
- `vision_only`: Nur Bilder (Schneller)

**Vision Backbones:**
- `resnet50`, `resnet101`
- `efficientnet_b0`, `efficientnet_b3`

**Text Models:**
- `distilbert-base-uncased` (Schnell, empfohlen)
- `bert-base-uncased` (Größer, genauer)

### 4. Model Config speichern

Erstelle `model_config.json` für Inference:

```json
{
  "num_classes": 36,
  "model_type": "multimodal",
  "vision_backbone": "resnet50",
  "text_model": "distilbert-base-uncased",
  "numeric_input_dim": 15,
  "use_vision": true,
  "use_text": true,
  "use_numeric": true,
  "label_classes": ["CLASS-1250", "CLASS-1274", "CLASS-1376", ...]
}
```

### 5. Inference (End-to-End)

Klassifiziere neue Rechnungen:

```bash
python predict_invoice_pytorch.py \
  --image batch1-0001.jpg \
  --model pytorch_models/best_model.pth \
  --config pytorch_models/model_config.json \
  --ocr-engine easyocr \
  --save-json
```

**Output:**
```
Verarbeite: batch1-0001.jpg
  [1/4] OCR Extraktion...
  [2/4] Feature-Extraktion...
  [3/4] Bild-Preprocessing...
  [4/4] Model Inference...

============================================================
ERGEBNIS:
============================================================
📊 Kategorie: CLASS-1963
✓ Confidence: 99.31%

Top 3 Predictions:
  1. CLASS-1963: 99.31%
  2. CLASS-1721: 0.18%
  3. CLASS-2038: 0.12%

✓ JSON gespeichert: batch1-0001_extracted.json
```

## 📊 Model Architektur

### Multimodal Model

```
Input:
├── Bild (3, 224, 224)      → Vision Encoder (ResNet50)     → 512-dim
├── Text (max_len=128)       → Text Encoder (DistilBERT)    → 512-dim
└── Numeric Features (15)    → Numeric Encoder (MLP)        → 128-dim

                              ↓ Concatenate (1152-dim)

                         Fusion Layer (MLP)
                              ↓ (256-dim)

                      Classifier (num_classes)
                              ↓
                         Softmax Probabilities
```

### Performance

Mit **Multimodal Approach**:
- **Baseline (Logistic Regression)**: 72% F1-Score
- **Random Forest**: 99% F1-Score
- **PyTorch Multimodal**: **99%+ F1-Score** (erwartet)

## 🎯 Verwendungsbeispiele

### Beispiel 1: Nur mit vorhandenen JSON-Daten

```python
from json_parser import InvoiceJSONParser
from invoice_model import create_model
import torch

# JSON parsen
parser = InvoiceJSONParser()
features = parser.extract_all_features(json_data)

# Model laden und prediction
model = create_model(num_classes=36, model_type='multimodal')
# ... (siehe predict_invoice_pytorch.py)
```

### Beispiel 2: OCR + Klassifikation

```python
from ocr_extractor import InvoiceOCRExtractor
from predict_invoice_pytorch import InvoicePredictor

# End-to-End
predictor = InvoicePredictor(
    model_path='pytorch_models/best_model.pth',
    model_config=config
)

result = predictor.predict_from_image('invoice.jpg')
print(f"Kategorie: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔍 Troubleshooting

### GPU nicht verfügbar
```bash
# CPU verwenden
python train_pytorch.py ... --device cpu
```

### EasyOCR Installation Probleme
```bash
# Alternative: Tesseract verwenden
python ocr_extractor.py --engine tesseract ...

# Oder EasyOCR manuell installieren
pip install easyocr --upgrade
```

### Speicher-Probleme beim Training
```bash
# Kleinere Batch-Size verwenden
python train_pytorch.py ... --batch-size 8

# Kleineres Model verwenden
python train_pytorch.py ... --model-type vision_only --vision-backbone resnet50
```

## 📈 Nächste Schritte

1. **Hyperparameter Tuning**: Learning Rate, Batch Size, Model Architecture
2. **Data Augmentation**: Mehr Bild-Transformationen für bessere Generalisierung
3. **Vision Transformer (ViT)**: Modernere Architektur für Bildverarbeitung
4. **Ensemble**: Kombiniere mehrere Models für höhere Accuracy
5. **API Deployment**: Flask/FastAPI für Production-Serving

## 📚 Literatur

- **ResNet**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **EasyOCR**: [GitHub Repository](https://github.com/JaidedAI/EasyOCR)
- **PyTorch**: [Official Documentation](https://pytorch.org/docs/)

---

**Autor**: Claude
**Datum**: 2025
**Version**: 1.0
