# ⚽ FIFA Scout AI — World Cup 2026 Edition

> **Classify any football player's tactical position from their stats alone.**  
> An end-to-end Machine Learning pipeline trained on 17,954 FIFA players, wrapped in a live interactive demo.

---

## Pipeline Overview

```
Raw FIFA Data (17,954 × 51)
        │
        ▼
   Data Cleaning          → 890 raw positions → 4 tactical classes (GK / DEF / MID / FWD)
        │
        ▼
   Feature Engineering    → 51 attributes → 34 curated features, IQR outlier capping
        │
        ▼
   Preprocessing          → StandardScaler + PCA (95 % variance → ~15 components)
        │
        ▼
   Model Training         → KNN · SVM (RBF) · Decision Tree
        │
        ▼
   Evaluation             → Confusion matrix, precision, recall, F1 per class
        │
        ▼
   Live Demo              → Search famous players or build your own to predict position
```

---

## Results

| Model         | Test Accuracy | GK F1 | DEF F1 | MID F1 | FWD F1 |
|---------------|:------------:|:-----:|:------:|:------:|:------:|
| **SVM (RBF)** | **81.2 %**   | 1.00  | 0.84   | 0.78   | 0.71   |
| KNN (k=5)     | 77.4 %       | 1.00  | 0.80   | 0.75   | 0.62   |
| Decision Tree | 75.3 %       | 1.00  | 0.78   | 0.73   | 0.61   |

**Key insight:** GKs are always classified perfectly — their physical profile is uniquely distinct.  
The hardest challenge is separating MID from FWD, which both require high technical skill.

---

## Quick Start

```bash
# 1. Create & activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the demo
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Demo Navigation

| Step | Screen | What It Shows |
|------|--------|---------------|
| ⚽ KICKOFF | Dataset Overview | 17,954 players, top-rated, nationality map |
| 🔍 SCOUTING | Exploratory Analysis | Radar by position, scatter separability |
| 🔄 TRANSFER | Data Cleaning | Missing values audit, 890 → 4 position map |
| 🏋️ TRAINING | Feature Engineering | Feature groups, outlier detection & capping |
| 🎯 TACTICS | Scaling & PCA | Standardization, cumulative variance, 2D PCA |
| 🏆 MATCH DAY | Model Training | Confusion matrices, per-class metrics for each model |
| 📊 FINAL WHISTLE | Results | Podium, accuracy comparison, F1 per class |
| ⭐ STAR SCOUT | Live Predictor | Search real players **or** build a custom profile |

---

## Dataset

`data/fifa_players.csv` — 17,954 players · 51 attributes · sourced from FIFA's official database.  
Covers crossing, finishing, dribbling, defensive stats, physical measurements, and more.

---

## Tech Stack

- **Data** — pandas, numpy  
- **ML** — scikit-learn (KNN, SVM, Decision Tree, PCA, StandardScaler)  
- **Visualization** — Plotly  
- **Demo App** — Streamlit  
