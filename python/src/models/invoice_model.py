"""
PyTorch Multimodal Model für Rechnungsklassifizierung
Kombiniert Vision (ResNet/EfficientNet) + Text (BERT) + Numerische Features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer


class VisionEncoder(nn.Module):
    """
    Vision Encoder (ResNet/EfficientNet für Rechnungsbilder)
    """

    def __init__(self, backbone='resnet50', pretrained=True, feature_dim=512):
        """
        Args:
            backbone: 'resnet50', 'resnet101', 'efficientnet_b0', etc.
            pretrained: Vortrainierte Weights verwenden
            feature_dim: Dimension der Output-Features
        """
        super(VisionEncoder, self).__init__()

        self.backbone_name = backbone

        if 'resnet' in backbone:
            # ResNet
            if backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unbekannter Backbone: {backbone}")

            # Entferne letzte FC-Layer
            backbone_out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif 'efficientnet' in backbone:
            # EfficientNet
            if backbone == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
            elif backbone == 'efficientnet_b3':
                self.backbone = models.efficientnet_b3(pretrained=pretrained)
            else:
                raise ValueError(f"Unbekannter Backbone: {backbone}")

            # Entferne letzte Classifier
            backbone_out_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unbekannter Backbone: {backbone}")

        # Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Bild Tensor (B, 3, H, W)

        Returns:
            Features (B, feature_dim)
        """
        features = self.backbone(x)
        features = self.projection(features)
        return features


class TextEncoder(nn.Module):
    """
    Text Encoder (BERT/DistilBERT für JSON-Text)
    """

    def __init__(self, model_name='distilbert-base-uncased', feature_dim=512, max_length=256):
        """
        Args:
            model_name: HuggingFace Model Name
            feature_dim: Dimension der Output-Features
            max_length: Maximale Sequenzlänge
        """
        super(TextEncoder, self).__init__()

        self.model_name = model_name
        self.max_length = max_length

        # BERT Model
        self.bert = AutoModel.from_pretrained(model_name)

        # Tokenizer (wird später in Training verwendet)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # BERT Output Dimension
        bert_dim = self.bert.config.hidden_size

        # Projection Layer
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_dim = feature_dim

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: Token IDs (B, max_length)
            attention_mask: Attention Mask (B, max_length)

        Returns:
            Features (B, feature_dim)
        """
        # BERT Forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # CLS Token als Sentence Embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Projection
        features = self.projection(cls_embedding)
        return features


class NumericEncoder(nn.Module):
    """
    Encoder für numerische Features
    """

    def __init__(self, input_dim, feature_dim=128):
        """
        Args:
            input_dim: Anzahl numerischer Input-Features
            feature_dim: Dimension der Output-Features
        """
        super(NumericEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: Numerische Features (B, input_dim)

        Returns:
            Features (B, feature_dim)
        """
        return self.encoder(x)


class MultimodalInvoiceClassifier(nn.Module):
    """
    Multimodal Classifier: Vision + Text + Numeric Features
    """

    def __init__(
        self,
        num_classes,
        vision_backbone='resnet50',
        text_model='distilbert-base-uncased',
        numeric_input_dim=15,
        vision_feature_dim=512,
        text_feature_dim=512,
        numeric_feature_dim=128,
        fusion_dim=256,
        use_vision=True,
        use_text=True,
        use_numeric=True
    ):
        """
        Args:
            num_classes: Anzahl Zielklassen
            vision_backbone: Vision Encoder Backbone
            text_model: Text Encoder Model
            numeric_input_dim: Anzahl numerischer Input-Features
            vision_feature_dim: Vision Feature Dimension
            text_feature_dim: Text Feature Dimension
            numeric_feature_dim: Numeric Feature Dimension
            fusion_dim: Fusion Layer Dimension
            use_vision: Vision Branch verwenden
            use_text: Text Branch verwenden
            use_numeric: Numeric Branch verwenden
        """
        super(MultimodalInvoiceClassifier, self).__init__()

        self.use_vision = use_vision
        self.use_text = use_text
        self.use_numeric = use_numeric

        # Encoders
        if use_vision:
            self.vision_encoder = VisionEncoder(
                backbone=vision_backbone,
                pretrained=True,
                feature_dim=vision_feature_dim
            )

        if use_text:
            self.text_encoder = TextEncoder(
                model_name=text_model,
                feature_dim=text_feature_dim
            )

        if use_numeric:
            self.numeric_encoder = NumericEncoder(
                input_dim=numeric_input_dim,
                feature_dim=numeric_feature_dim
            )

        # Berechne Fusion Input Dimension
        fusion_input_dim = 0
        if use_vision:
            fusion_input_dim += vision_feature_dim
        if use_text:
            fusion_input_dim += text_feature_dim
        if use_numeric:
            fusion_input_dim += numeric_feature_dim

        # Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classifier Head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, image=None, input_ids=None, attention_mask=None, numeric_features=None):
        """
        Args:
            image: Bild Tensor (B, 3, H, W) - optional
            input_ids: Token IDs (B, max_length) - optional
            attention_mask: Attention Mask (B, max_length) - optional
            numeric_features: Numerische Features (B, numeric_dim) - optional

        Returns:
            Logits (B, num_classes)
        """
        features_list = []

        # Vision Branch
        if self.use_vision and image is not None:
            vision_features = self.vision_encoder(image)
            features_list.append(vision_features)

        # Text Branch
        if self.use_text and input_ids is not None and attention_mask is not None:
            text_features = self.text_encoder(input_ids, attention_mask)
            features_list.append(text_features)

        # Numeric Branch
        if self.use_numeric and numeric_features is not None:
            numeric_feats = self.numeric_encoder(numeric_features)
            features_list.append(numeric_feats)

        # Concatenate Features
        if len(features_list) == 0:
            raise ValueError("Mindestens eine Modalität muss aktiviert sein!")

        combined_features = torch.cat(features_list, dim=1)

        # Fusion
        fused_features = self.fusion(combined_features)

        # Classification
        logits = self.classifier(fused_features)

        return logits


class VisionOnlyClassifier(nn.Module):
    """
    Vision-Only Classifier (einfacher, wenn nur Bilder verwendet werden)
    """

    def __init__(
        self,
        num_classes,
        backbone='resnet50',
        pretrained=True
    ):
        """
        Args:
            num_classes: Anzahl Zielklassen
            backbone: Vision Backbone
            pretrained: Vortrainierte Weights
        """
        super(VisionOnlyClassifier, self).__init__()

        # Vision Encoder
        self.vision_encoder = VisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            feature_dim=512
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image):
        """
        Args:
            image: Bild Tensor (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        features = self.vision_encoder(image)
        logits = self.classifier(features)
        return logits


def create_model(
    num_classes,
    model_type='multimodal',
    vision_backbone='resnet50',
    text_model='distilbert-base-uncased',
    numeric_input_dim=15,
    use_vision=True,
    use_text=True,
    use_numeric=True
):
    """
    Factory-Funktion zum Erstellen von Models

    Args:
        num_classes: Anzahl Klassen
        model_type: 'multimodal', 'vision_only', 'text_only'
        vision_backbone: Vision Encoder Backbone
        text_model: Text Encoder Model
        numeric_input_dim: Anzahl numerischer Features
        use_vision: Vision verwenden
        use_text: Text verwenden
        use_numeric: Numerische Features verwenden

    Returns:
        PyTorch Model
    """
    if model_type == 'multimodal':
        model = MultimodalInvoiceClassifier(
            num_classes=num_classes,
            vision_backbone=vision_backbone,
            text_model=text_model,
            numeric_input_dim=numeric_input_dim,
            use_vision=use_vision,
            use_text=use_text,
            use_numeric=use_numeric
        )
    elif model_type == 'vision_only':
        model = VisionOnlyClassifier(
            num_classes=num_classes,
            backbone=vision_backbone
        )
    else:
        raise ValueError(f"Unbekannter model_type: {model_type}")

    return model


def main():
    """
    Test der Models
    """
    print("="*60)
    print("MODEL TEST")
    print("="*60)

    # Dummy Daten
    batch_size = 4
    num_classes = 36

    # Images
    images = torch.randn(batch_size, 3, 224, 224)

    # Text (Token IDs)
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)

    # Numeric Features
    numeric_features = torch.randn(batch_size, 15)

    # Test 1: Multimodal Model
    print("\n1. Multimodal Model (Vision + Text + Numeric)")
    model = create_model(
        num_classes=num_classes,
        model_type='multimodal',
        use_vision=True,
        use_text=True,
        use_numeric=True
    )

    logits = model(
        image=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        numeric_features=numeric_features
    )
    print(f"   Output Shape: {logits.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test 2: Vision-Only Model
    print("\n2. Vision-Only Model")
    model_vision = create_model(
        num_classes=num_classes,
        model_type='vision_only'
    )

    logits_vision = model_vision(images)
    print(f"   Output Shape: {logits_vision.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_vision.parameters()):,}")

    print("\n✓ Model Tests erfolgreich!")


if __name__ == "__main__":
    main()
