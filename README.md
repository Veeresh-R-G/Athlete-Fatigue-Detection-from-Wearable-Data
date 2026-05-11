[Kaggle Notebook](https://www.kaggle.com/code/veeresh1104/athlete-fatigue-detection-from-wearable-data)

# 🏅 Athlete Fatigue Detection — GRU on Real Strava Data
> Detect personal fatigue and overtraining risk from real workout history  
> Built using Strava API v3 + PyTorch GRU on 729 days of real training data

---

## 📌 Problem Statement
Can we detect athlete fatigue and overtraining risk from training history alone,
using only data available from a consumer fitness app like Strava?

This is the core problem Runna's ML team solves at scale — adaptive training
plans require knowing whether an athlete is recovered, fatigued, or at
overtraining risk before prescribing the next session.

---

## 🏗️ System Architecture
```
Strava API v3 (OAuth 2.0)
↓
714 real activities (2022–2026)
↓
Training Load Feature Engineering
→ ACWR (Acute:Chronic Workload Ratio)
→ HR trend over 7 days
→ Load monotony
→ Rest day patterns
↓
ACWR-based fatigue labelling
→ Fresh (<0.8) | Optimal (0.8-1.3) | Fatigued (1.3-1.5) | Overtraining (>1.5)
↓
GRU on 7-day sliding windows
↓
Real-time fatigue prediction
```
---

## 📊 Dataset — Real Personal Strava Data
- **Source:** Strava API v3 (personal account)
- **Activities:** 714 cardio sessions (696 runs)
- **Date range:** August 2022 → May 2026
- **HR data:** 350 activities (from May 2024 when HR monitor acquired)
- **Final sequences:** 729 labelled 7-day windows

### Class Distribution:
| State | Days | % | ACWR Range |
|---|---|---|---|
| Fresh/Undertrained | 224 | 30.7% | < 0.8 |
| Optimal | 344 | 47.2% | 0.8 – 1.3 |
| Fatigued | 67 | 9.2% | 1.3 – 1.5 |
| Overtraining Risk | 94 | 12.9% | > 1.5 |

---

## 🔧 Feature Engineering (9 features per day)
| Feature | Description | Sports Science Basis |
|---|---|---|
| `daily_load` | Duration × HR intensity | Session RPE proxy |
| `avg_hr` | Mean heart rate | Cardiac stress indicator |
| `total_distance` | km per day | Volume metric |
| `hr_trend_7d` | Linear slope of HR | Cardiac drift = fatigue signal |
| `load_monotony` | Mean/std of load | Low variation = overtraining risk |
| `rest_days_7d` | Rest days in window | Recovery behaviour |
| `acute_load` | 7-day rolling sum | Short-term fatigue |
| `chronic_load` | 28-day rolling avg | Fitness/long-term adaptation |
| `acwr` | Acute / Chronic | Industry standard injury risk metric |

---

## 🧠 Model Architecture — GRU
```
Input: (batch, 7 days, 9 features)
↓
GRU Layer 1 (hidden=64) — daily pattern recognition
↓
GRU Layer 2 (hidden=64) — weekly trend recognition
↓
BatchNorm + Dropout(0.4)
↓
FC: 64 → 4 classes
↓
Fresh / Optimal / Fatigued / Overtraining Risk
**Why GRU over LSTM:**
7-day sequences are short — GRU's unified hidden state captures
weekly patterns with 25% fewer parameters than LSTM.
Validated empirically: same accuracy, faster training.
```
---

## 📈 Results — Real Data Performance
| Metric | Score |
|---|---|
| Test Accuracy | **98.3%** |
| Optimal F1 | 0.95 |
| Fresh F1 | 0.91 |
| Weighted avg F1 | 0.92 |

**Honest limitations:**
- Fatigued class: 0% recall (only 4 test samples — distribution shift)
- Test period (recent months) shows improved training discipline
  → fewer Fatigued/Overtraining days than training period
- ACWR-based labels are proxies, not ground truth
  (ground truth requires subjective RPE + blood lactate markers)

---

## 🔑 Key Engineering Decisions

**1. Chronological train/test split**
Time-series data must be split chronologically, never randomly.
Random split causes future data leaking into training — the model
would "know" future training patterns.

**2. SMOTE after split, on train only**
Fatigued (9%) and Overtraining (13%) are minority classes.
SMOTE applied only to training set to prevent synthetic samples
polluting evaluation.

**3. GRU over LSTM**
7-day sequences don't require LSTM's separate cell state.
GRU achieves equivalent accuracy with fewer parameters —
better for edge deployment (Apple Watch, Garmin).

**4. BatchNorm for real data stability**
Real wearable data has significantly more noise than synthetic.
BatchNorm stabilises training by normalising layer inputs.

**5. Gradient clipping**
`clip_grad_norm_(model.parameters(), 1.0)` prevents exploding
gradients which are common with real noisy time-series.

---

## 🏃 My Fatigue State Today(11th May 2026)
```
ACWR        : 0.84  (optimal zone: 0.8–1.3)
Acute load  : 108.9 AU
Chronic load: 129.9 AU
Prediction  : Fresh / Undertrained (93.1%)
Training below recent average — slight deload
Good time for a quality session or extra rest
```
---

## 🚀 Production Extensions
1. Integrate Garmin/Oura HRV data for resting HR + sleep signals
2. Add attention mechanism to highlight which days drove prediction
3. Deploy as FastAPI endpoint consumed by mobile app
4. Personalised baseline per athlete (fatigue is relative)
5. Webhook integration for real-time post-activity inference

---

## 🛠️ Tech Stack
- Python, NumPy, Pandas
- Strava API v3 (OAuth 2.0, token refresh)
- PyTorch (GRU, Dataset, DataLoader, BatchNorm)
- imbalanced-learn (SMOTE)
- Scikit-learn (StandardScaler, metrics)
- Matplotlib, Seaborn

---

## 📚 References
- Gabbett (2016) — ACWR and injury risk in team sport athletes
- Plews et al. (2013) — HRV and training monitoring
- Buchheit (2014) — Monitoring training status with HRV
- Strava API v3 Documentation
