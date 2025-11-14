"""
PyTorch Dataset für Rechnungsbilder + JSON
Multimodal Dataset: Bild + Text + Numerische Features
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.data.json_parser import InvoiceJSONParser


class InvoiceDataset(Dataset):
    """
    PyTorch Dataset für Rechnungen (Multimodal: Bild + JSON-Features)
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        labels_df: Optional[pd.DataFrame] = None,
        image_size: Tuple[int, int] = (224, 224),
        transform=None,
        mode='train'
    ):
        """
        Args:
            csv_path: Pfad zur CSV mit JSON-Daten (batch1_1.csv)
            image_dir: Verzeichnis mit Rechnungsbildern
            labels_df: DataFrame mit Labels (Inv_Id, Product_Category)
            image_size: Bildgröße für ResNet/ViT (default: 224x224)
            transform: Optionale Transformationen
            mode: 'train', 'val', 'test'
        """
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.mode = mode

        # CSV laden
        print(f"Lade Dataset: {csv_path}")
        self.df_csv = pd.read_csv(csv_path)
        print(f"  Zeilen: {len(self.df_csv)}")

        # JSON Parser
        self.json_parser = InvoiceJSONParser()

        # Features extrahieren
        print("Extrahiere Features aus JSON...")
        self.features_df = self.json_parser.load_from_csv(csv_path)

        # Labels (falls vorhanden)
        self.labels_df = labels_df
        self.has_labels = labels_df is not None

        if self.has_labels:
            # Merge mit Labels basierend auf File Name / Inv_Id
            # Hier müssen wir File Name zu Inv_Id mappen
            # Annahme: File Name kann zu Inv_Id gemappt werden (z.B. über separate Mapping-Datei)
            print("Lade Labels...")
            self.label_encoder = None
            self.num_classes = labels_df['Product_Category'].nunique()
            print(f"  Klassen: {self.num_classes}")

            # Label Encoding
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(labels_df['Product_Category'])
        else:
            self.labels = None
            self.num_classes = 0

        # Image Transforms
        if transform is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize((image_size[0] + 32, image_size[1] + 32)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform

        # Numerische Feature-Namen
        self.numeric_feature_names = [
            'total_quantity', 'total_items_price', 'avg_item_price', 'num_items',
            'tax', 'discount', 'total', 'tax_rate', 'discount_rate',
            'invoice_year', 'invoice_month', 'invoice_day', 'invoice_weekday',
            'invoice_quarter', 'days_until_due'
        ]

        print(f"✓ Dataset initialisiert: {len(self)} Samples")

    def __len__(self):
        return len(self.df_csv)

    def __getitem__(self, idx):
        """
        Gibt ein Sample zurück

        Returns:
            dict mit:
                - image: Tensor (3, H, W)
                - text: String (combined_text)
                - numeric_features: Tensor (n_features,)
                - label: int (falls vorhanden)
                - file_name: str
        """
        # File Name
        file_name = self.df_csv.iloc[idx]['File Name']

        # Bild laden
        image_path = self.image_dir / file_name
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Warnung: Bild nicht gefunden {file_name}, verwende Dummy")
            # Dummy-Bild erstellen
            image = torch.zeros(3, self.image_size[0], self.image_size[1])

        # Text Features
        text = self.features_df.iloc[idx]['combined_text']

        # Numerische Features
        numeric_values = []
        for feature_name in self.numeric_feature_names:
            if feature_name in self.features_df.columns:
                value = self.features_df.iloc[idx][feature_name]
                # NaN zu 0 konvertieren
                if pd.isna(value):
                    value = 0.0
                numeric_values.append(float(value))
            else:
                numeric_values.append(0.0)

        numeric_features = torch.tensor(numeric_values, dtype=torch.float32)

        # Sample dict
        sample = {
            'image': image,
            'text': text,
            'numeric_features': numeric_features,
            'file_name': file_name
        }

        # Label (falls vorhanden)
        if self.has_labels and self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return sample

    def get_text_features_list(self):
        """Gibt alle Text-Features als Liste zurück (für BERT-Tokenizer)"""
        return self.features_df['combined_text'].tolist()

    def get_numeric_feature_dim(self):
        """Gibt Anzahl numerischer Features zurück"""
        return len(self.numeric_feature_names)


def create_dataloaders(
    train_csv,
    train_image_dir,
    val_csv=None,
    val_image_dir=None,
    labels_df=None,
    batch_size=16,
    num_workers=4,
    image_size=(224, 224)
):
    """
    Erstellt Train/Val DataLoaders

    Args:
        train_csv: Pfad zur Training CSV
        train_image_dir: Verzeichnis mit Training-Bildern
        val_csv: Pfad zur Validation CSV (optional)
        val_image_dir: Verzeichnis mit Validation-Bildern (optional)
        labels_df: DataFrame mit Labels
        batch_size: Batch Size
        num_workers: Anzahl Worker für DataLoader
        image_size: Bildgröße

    Returns:
        train_loader, val_loader (oder nur train_loader falls kein Val)
    """
    # Training Dataset
    train_dataset = InvoiceDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        labels_df=labels_df,
        image_size=image_size,
        mode='train'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Validation Dataset (falls vorhanden)
    if val_csv and val_image_dir:
        val_dataset = InvoiceDataset(
            csv_path=val_csv,
            image_dir=val_image_dir,
            labels_df=labels_df,
            image_size=image_size,
            mode='val'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    return train_loader, None


def main():
    """
    Test des Datasets
    """
    # Beispiel: batch_1 Daten
    csv_path = "../Data/batch_1/batch_1/batch1_1.csv"
    image_dir = "../Data/batch_1/batch_1/batch1_1"

    # Dataset erstellen (ohne Labels für Test)
    dataset = InvoiceDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        labels_df=None,  # Keine Labels für Test
        image_size=(224, 224),
        mode='test'
    )

    print(f"\nDataset Länge: {len(dataset)}")
    print(f"Numerische Features: {dataset.get_numeric_feature_dim()}")

    # Erstes Sample
    sample = dataset[0]

    print("\n" + "="*60)
    print("SAMPLE 0:")
    print("="*60)
    print(f"File Name: {sample['file_name']}")
    print(f"Image Shape: {sample['image'].shape}")
    print(f"Text (Auszug): {sample['text'][:100]}...")
    print(f"Numeric Features Shape: {sample['numeric_features'].shape}")
    print(f"Numeric Features: {sample['numeric_features'][:5]}")

    # DataLoader Test
    print("\n" + "="*60)
    print("DATALOADER TEST:")
    print("="*60)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch = next(iter(dataloader))
    print(f"Batch Images Shape: {batch['image'].shape}")
    print(f"Batch Texts: {len(batch['text'])} Texte")
    print(f"Batch Numeric Features Shape: {batch['numeric_features'].shape}")


if __name__ == "__main__":
    main()
