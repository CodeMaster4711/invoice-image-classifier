# TODO: Invoice Image Classifier

## Phase 1: Setup & Datenexploration
- [ ] Virtual Environment aktivieren und Dependencies installieren
  ```bash
  cd python
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- [ ] Jupyter Notebook erstellen für Datenexploration
- [ ] Train.csv laden und erste Analyse durchführen
  - Anzahl der Zeilen/Spalten
  - Anzahl der verschiedenen Product_Categories
  - Verteilung der Kategorien (Balkendiagramm)
  - Null-Werte prüfen
- [ ] Item_Description Texte analysieren
  - Durchschnittliche Textlänge
  - Häufigste Wörter
  - Beispiele aus verschiedenen Kategorien anschauen

## Phase 2: Datenvorverarbeitung
- [ ] Text-Cleaning Funktion erstellen
  - Lowercase
  - Sonderzeichen entfernen
  - Stopwords entfernen (optional)
- [ ] Features vorbereiten
  - Vendor_Code enkodieren
  - GL_Code enkodieren
  - Inv_Amt normalisieren
- [ ] Train/Validation Split (80/20)
- [ ] Label Encoding für Product_Category

## Phase 3: Feature Engineering
- [ ] TF-IDF Vectorizer für Item_Description implementieren
  - Max features festlegen (z.B. 5000)
  - N-grams testen (1,2)
- [ ] One-Hot-Encoding für kategorische Features
- [ ] Feature Matrix zusammenbauen

## Phase 4: Baseline Model
- [ ] Logistic Regression trainieren
- [ ] Predictions auf Validation Set
- [ ] Evaluation Metriken berechnen
  - Accuracy
  - F1-Score (weighted)
  - Confusion Matrix
- [ ] Ergebnisse dokumentieren

## Phase 5: Advanced Models
- [ ] Random Forest Classifier
  - Hyperparameter tuning (n_estimators, max_depth)
  - Feature Importance analysieren
- [ ] XGBoost Classifier
  - Hyperparameter tuning
  - Cross-Validation
- [ ] Optional: Neural Network (PyTorch)
  - Embeddings für Text
  - Multi-layer Perceptron
- [ ] Bestes Model auswählen

## Phase 6: Test & Submission
- [ ] Bestes Model auf vollständigem Train.csv trainieren
- [ ] Predictions auf Test.csv generieren
- [ ] Submission File erstellen (gemäß sample_submission.csv Format)
- [ ] Ergebnisse validieren

## Phase 7: Dokumentation & Cleanup
- [ ] README.md aktualisieren mit
  - Projektbeschreibung
  - Setup Anleitung
  - Ergebnisse
  - Nächste Schritte
- [ ] Code aufräumen und kommentieren
- [ ] .gitignore anpassen (Data/, .venv/, etc.)

## Notizen
- Dataset Location: `Data/Dataset/`
- Train.csv: ~931KB, Test.csv: ~381KB
- Kategorien: CLASS-XXXX Format
- Main Features: Item_Description (Text), Vendor_Code, GL_Code, Inv_Amt
