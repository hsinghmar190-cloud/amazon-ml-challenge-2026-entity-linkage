# Amazon ML Challenge 2026: Multi-Source Entity Linkage System

An end-to-end, high-precision machine learning pipeline designed to resolve high-cardinality entity variations across 1.73M queries and 9.97M targets under strict Macro F0.5 evaluation constraints.

---

## 🏆 Key Highlights & Benchmarks
* **Final Locked Score:** 0.213 on the challenge leaderboard
* **Candidate Blocking Recall:** 90.13% True Positives captured (6.88M / 7.63M Ground Truth pairs)
* **Model Logloss:** 0.0926 early stopping convergence (Round 73)
* **Memory Optimization:** Reduced peak RAM usage from 6.2 GB+ to under 1.8 GB using a Zero-Cache Streaming Inference Engine

---

## 🏗️ Architecture Overview

1. **High-Recall Inverted Indexing & Blocking:**
   - Multi-source candidate generation across Source 1, Source 2, and Source 3.
   - Regex-based Indian postal PIN extraction (`\b[1-9][0-9]{5}\b`) combined with rare-token inverted hashing to filter the 17M+ search space.

2. **9-Dimensional Feature Space:**
   - Token Jaccard overlap and 3-to-5 character n-gram cosine similarities.
   - Inverse Document Frequency (IDF) rarity weighting to downweight corporate stopwords (*pvt*, *ltd*, *enterprises*).
   - Ternary geospatial PIN validation (`+1` match, `-1` mismatch, `0` missing/unobserved).
   - Address missingness indicator flags to dynamically balance scoring regimes.

3. **Monotonic LightGBM Ranker:**
   - Enforced directional monotonicity constraints (`[1, 1, 1, 1, 1, 1, 1, -1, 0]`) across features to eliminate artificial score drops on higher similarity.
   - Applied a 4.0x False Positive asymmetry penalty (`scale_pos_weight = 0.25`) specifically tuned for the precision-heavy Macro F0.5 metric.

4. **Zero-Cache Streaming Inference:**
   - Direct disk-to-disk line streaming processing candidates on the fly.
   - Strict 1-to-1 bipartite target mapping to prevent cross-source cluster leakage.

---

## 👥 Engineering Team
* **Himanshu Verma:** ML Engineering Lead (Feature engineering, Monotonic LightGBM ranker, Zero-Cache streaming inference engine)
* **Dheeraj:** Data Engineering (Inverted index candidate generation & high-speed blocking)
* **Arnav:** Quality Assurance & Validation (Candidate recall audit and schema compliance)
* **Jyoti:** Technical Documentation & Approach Statement Synthesis

---

## ⚙️ Repository Structure
```text
├── notebooks/
│   ├── 87_v8_lightgbm_trainer.py   # Model training with monotonic constraints
│   ├── 90_v8_test_inference.py     # Zero-cache streaming test inference
│   └── blocking.py                 # Candidate indexing & blocking
├── requirements.txt                # Core dependencies
└── README.md                       # System documentation

🚀 Reproduction Pipeline

# 1. Install dependencies
pip install -r requirements.txt

# 2. Candidate generation
python notebooks/blocking.py

# 3. Model training
python notebooks/87_v8_lightgbm_trainer.py

# 4. Final test streaming inference
python notebooks/90_v8_test_inference.py
