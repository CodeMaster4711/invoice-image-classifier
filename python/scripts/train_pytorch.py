"""
PyTorch Training Pipeline für Rechnungsklassifizierung
Trainiert Multimodal Model (Vision + Text + Numeric)
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import InvoiceDataset
from src.models.invoice_model import create_model
from sklearn.metrics import accuracy_score, f1_score, classification_report


class InvoiceTrainer:
    """
    Trainer für Rechnungsklassifizierung
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_classes,
        device='cuda',
        learning_rate=1e-4,
        weight_decay=1e-5,
        label_encoder=None,
        tokenizer=None,
        max_length=128
    ):
        """
        Args:
            model: PyTorch Model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_classes: Anzahl Klassen
            device: 'cuda' oder 'cpu'
            learning_rate: Learning Rate
            weight_decay: Weight Decay für AdamW
            label_encoder: LabelEncoder für Klassen
            tokenizer: HuggingFace Tokenizer für Text
            max_length: Maximale Sequenzlänge für Text
        """
        self.device = device  # Zuerst als self.device speichern
        self.model = model.to(self.device)  # Dann verwenden
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.label_encoder = label_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.best_f1 = 0.0

    def tokenize_texts(self, texts):
        """
        Tokenisiert Text-Batch

        Args:
            texts: Liste von Texten

        Returns:
            input_ids, attention_mask (Tensors)
        """
        if self.tokenizer is None:
            return None, None

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded['input_ids'].to(self.device), encoded['attention_mask'].to(self.device)

    def train_epoch(self):
        """
        Trainiert eine Epoche

        Returns:
            Durchschnittlicher Loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc='Training')

        for batch in pbar:
            # Daten
            images = batch['image'].to(self.device)
            numeric_features = batch['numeric_features'].to(self.device)
            labels = batch['label'].to(self.device)

            # Text tokenisieren
            texts = batch['text']
            input_ids, attention_mask = self.tokenize_texts(texts)

            # Forward Pass
            self.optimizer.zero_grad()

            logits = self.model(
                image=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_features=numeric_features
            )

            # Loss
            loss = self.criterion(logits, labels)

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            # Tracking
            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Validierung

        Returns:
            avg_loss, accuracy, f1_score
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')

            for batch in pbar:
                # Daten
                images = batch['image'].to(self.device)
                numeric_features = batch['numeric_features'].to(self.device)
                labels = batch['label'].to(self.device)

                # Text tokenisieren
                texts = batch['text']
                input_ids, attention_mask = self.tokenize_texts(texts)

                # Forward Pass
                logits = self.model(
                    image=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numeric_features=numeric_features
                )

                # Loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Predictions
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix({'loss': loss.item()})

        # Metriken
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1

    def train(self, num_epochs, save_dir='models'):
        """
        Trainiert das Model

        Args:
            num_epochs: Anzahl Epochen
            save_dir: Verzeichnis zum Speichern der Checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("="*60)
        print("TRAINING START")
        print("="*60)
        print(f"Epochen: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 60)

            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation
            val_loss, val_acc, val_f1 = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Val F1-Score: {val_f1:.4f}")

            # Learning Rate Scheduler
            self.scheduler.step(val_f1)

            # Save Best Model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                checkpoint_path = save_dir / 'best_model.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_f1)
                print(f"✓ Bestes Model gespeichert: {checkpoint_path}")

            # Save Regular Checkpoint
            if epoch % 5 == 0:
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(checkpoint_path, epoch, val_f1)

        print("\n" + "="*60)
        print("TRAINING ABGESCHLOSSEN")
        print("="*60)
        print(f"Bester F1-Score: {self.best_f1:.4f}")

    def save_checkpoint(self, path, epoch, f1_score):
        """Speichert Model Checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'f1_score': f1_score,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'val_f1_scores': self.val_f1_scores
        }

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Lädt Model Checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_accuracies = checkpoint['val_accuracies']
        self.val_f1_scores = checkpoint['val_f1_scores']

        print(f"✓ Checkpoint geladen: {path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  F1-Score: {checkpoint['f1_score']:.4f}")


def main():
    """
    Haupt-Training Pipeline
    """
    parser = argparse.ArgumentParser(description="PyTorch Training für Rechnungsklassifizierung")

    # Daten
    parser.add_argument('--train-csv', type=str, required=True, help='Pfad zur Training CSV mit JSON')
    parser.add_argument('--train-images', type=str, required=True, help='Verzeichnis mit Training-Bildern')
    parser.add_argument('--labels-csv', type=str, required=True, help='Pfad zur Labels CSV (Inv_Id, Product_Category)')

    # Model
    parser.add_argument('--model-type', type=str, default='multimodal', choices=['multimodal', 'vision_only'])
    parser.add_argument('--vision-backbone', type=str, default='resnet50')
    parser.add_argument('--text-model', type=str, default='distilbert-base-uncased')

    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--val-split', type=float, default=0.2)

    # Misc
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, mps, cpu, or auto')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='pytorch_models')

    args = parser.parse_args()

    print("="*60)
    print("SETUP")
    print("="*60)

    # Device Selection (M3 GPU Support)
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🚀 Verwende CUDA GPU")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🚀 Verwende Apple M3 GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("⚠️  Verwende CPU (GPU nicht verfügbar)")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    # Labels laden
    print(f"Lade Labels: {args.labels_csv}")
    labels_df = pd.read_csv(args.labels_csv)
    num_classes = labels_df['Product_Category'].nunique()
    print(f"Klassen: {num_classes}")

    # Dataset
    print(f"Lade Dataset: {args.train_csv}")
    full_dataset = InvoiceDataset(
        csv_path=args.train_csv,
        image_dir=args.train_images,
        labels_df=labels_df,
        image_size=(224, 224),
        mode='train'
    )

    # Train/Val Split
    val_size = int(args.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train Size: {len(train_dataset)}")
    print(f"Val Size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print(f"Erstelle Model: {args.model_type}")
    model = create_model(
        num_classes=num_classes,
        model_type=args.model_type,
        vision_backbone=args.vision_backbone,
        text_model=args.text_model,
        numeric_input_dim=full_dataset.get_numeric_feature_dim()
    )

    # Tokenizer (für Text)
    tokenizer = None
    if args.model_type == 'multimodal':
        tokenizer = AutoTokenizer.from_pretrained(args.text_model)

    # Trainer (device ist bereits torch.device Objekt)
    trainer = InvoiceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,  # torch.device Objekt übergeben
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_encoder=full_dataset.label_encoder,
        tokenizer=tokenizer,
        max_length=128
    )

    # Training
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

    print("\n✓ Training abgeschlossen!")
    print(f"Beste Modelle gespeichert in: {args.save_dir}")


if __name__ == "__main__":
    main()
