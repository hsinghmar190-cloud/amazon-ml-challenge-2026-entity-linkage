# Amazon ML Challenge 2026: Multi-Source Entity Linkage System

An end-to-end, high-precision machine learning pipeline built to solve high-cardinality multi-source entity linkage across 1.73M queries and 9.97M targets under strict Macro F0.5 constraints.

---

## 🏆 Key Achievements
* **Locked Benchmark Score:** 0.213 on the evaluation leaderboard
* **Candidate Blocking Recall:** 90.13% true positives captured (6.88M / 7.63M ground truth pairs)
* **Model Logloss:** 0.0926 early-stopping convergence
* **Memory Optimization:** Kept peak RAM usage under 1.8 GB via Zero-Cache Streaming Inference (preventing 6.2 GB+ system thrashing)

---

## 🏗️ Architecture Overview

The pipeline operates in four coordinated stages:

1. **High-Recall Inverted Indexing & Blocking (`blocking.py`):**
   * Multi-source candidate generation across Source 1, Source 2, and Source 3.
   * Regex-based Indian postal PIN extraction (`\b[1-9][0-9]{5}\b`) and rare-token inverted hashing to filter the 17M+ search space.

2. **9-Dimensional Feature Extraction:**
   * Token Jaccard overlap and 3-to-5 character n-gram cosine similarities.
   * Inverse Document Frequency (IDF) rarity weighting to downweight corporate stopwords (*pvt*, *ltd*, *enterprises*).
   * Ternary geospatial PIN validation (`+1` match, `-1` mismatch, `0` missing).
   * Address missingness indicator flags to toggle scoring regimes.

3. **Monotonic LightGBM Ranker (`87_v8_lightgbm_trainer.py`):**
   * Enforced strict directional monotonicity constraints across all lexical and geospatial features.
   * Configured a 4.0x False Positive asymmetry penalty to directly optimize the precision-heavy Macro F0.5 metric.

4. **Zero-Cache Streaming Inference Engine (`90_v8_test_inference.py`):**
   * Streamlined, disk-to-disk line processing evaluating candidate pairs on the fly.
   * 1-to-1 bipartite target mapping to prevent cross-source cluster leakage.

---

## 👥 Engineering Team
* **Himanshu Verma:** ML Engineering Lead (Feature engineering, Monotonic LightGBM modeling, Zero-Cache streaming engine)
* **Dheeraj:** Data Engineering (Inverted index candidate generation & high-speed blocking)
* **Arnav:** Quality Assurance & Validation (Candidate recall audit and schema compliance)
* **Jyoti:** Technical Documentation & Approach Statement Synthesis

---

## ⚙️ How to Run

### 1. Requirements
```bash
pip install -r requirements.txt
