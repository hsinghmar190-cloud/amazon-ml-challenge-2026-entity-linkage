import os
import re
import csv
import math
import time
import random
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# ==============================================================================
# PATHS
# ==============================================================================
TRAIN_DIR = os.path.join("dataset", "train")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "lgbm_v8_model.txt")

# ==============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ==============================================================================
STOPWORDS = {'pvt', 'ltd', 'limited', 'private', 'enterprises', 'corp', 'co', 'llp', 'inc'}

def clean_text(text):
    if not text: return ""
    return re.sub(r'[^a-z0-9\s]', ' ', str(text).lower()).strip()

def extract_pin(text):
    if not text: return ""
    m = re.search(r'\b[1-9][0-9]{5}\b', str(text))
    return m.group(0) if m else ""

def get_tokens(text):
    clean = clean_text(text)
    return [t for t in clean.split() if t not in STOPWORDS and len(t) > 2]

def char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9]', '', str(text).lower())
    if len(clean) < n: return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def cosine_similarity(set_a, set_b):
    if not set_a or not set_b: return 0.0
    inter = len(set_a.intersection(set_b))
    norm = math.sqrt(len(set_a)) * math.sqrt(len(set_b))
    return inter / norm if norm > 0 else 0.0

def extract_9_features(s1_name, s1_addr, t_name, t_addr, idf_dict):
    s1_toks = set(get_tokens(s1_name))
    t_toks = set(get_tokens(t_name))
    s1_pin = extract_pin(s1_addr)
    t_pin = extract_pin(t_addr)
    
    inter = s1_toks.intersection(t_toks)
    union = s1_toks.union(t_toks)
    f1_jaccard = len(inter) / len(union) if union else 0.0
    
    f2_cos = cosine_similarity(char_ngrams(s1_name, 3), char_ngrams(t_name, 3))
    
    s1_sort = " ".join(sorted(list(s1_toks)))
    t_sort = " ".join(sorted(list(t_toks)))
    f3_sort_ratio = SequenceMatcher(None, s1_sort, t_sort).ratio() if (s1_sort and t_sort) else 0.0
    
    len1, len2 = len(s1_toks), len(t_toks)
    f4_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    
    shared_idfs = [idf_dict.get(t, 1.0) for t in inter]
    f5_max_idf = max(shared_idfs) if shared_idfs else 0.0
    f6_sum_idf = sum(shared_idfs)
    
    f7_pin_exact = 1 if (s1_pin and t_pin and s1_pin == t_pin) else 0
    f8_pin_mismatch = 1 if (s1_pin and t_pin and s1_pin != t_pin) else 0
    f9_addr_missing = 1 if (not s1_addr.strip() or not t_addr.strip()) else 0
    
    return [f1_jaccard, f2_cos, f3_sort_ratio, f4_len_ratio, f5_max_idf, f6_sum_idf, f7_pin_exact, f8_pin_mismatch, f9_addr_missing]

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    print("=" * 75)
    print("--- [V8 PHASE 3: STANDALONE HIGH-SPEED LIGHTGBM TRAINER] ---")
    print("=" * 75)
    t0 = time.time()

    # 1. Load Entity Metadata
    print("[1/5] Loading Entity Metadata into RAM...")
    s1_meta = {}
    with open(S1_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for r in reader:
            eid = (r.get('source1_entity_id') or r.get('entity_id') or '').strip()
            if eid:
                s1_meta[eid] = (r.get('business_name', ''), r.get('business_address', ''))

    target_meta = {}
    for p, prefix in [(S2_PATH, "s2"), (S3_PATH, "s3")]:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for r in reader:
                    eid = (r.get(f'{prefix}_entity_id') or r.get('entity_id') or '').strip()
                    if eid:
                        target_meta[eid] = (r.get('business_name', ''), r.get('business_address', ''))

    print(f"Loaded {len(s1_meta):,} S1 entities | {len(target_meta):,} Target entities.")
    target_ids = list(target_meta.keys())

    # 2. Build Global IDF Dictionary
    print("[2/5] Building Global IDF Dictionary...")
    doc_freq = defaultdict(int)
    total_docs = len(s1_meta) + len(target_meta)
    for name, _ in s1_meta.values():
        for t in set(get_tokens(name)): doc_freq[t] += 1
    for name, _ in target_meta.values():
        for t in set(get_tokens(name)): doc_freq[t] += 1

    idf_dict = {t: math.log((1 + total_docs) / (1 + df)) + 1.0 for t, df in doc_freq.items()}
    del doc_freq

    # 3. Sample 100,000 Positives
    print("[3/5] Sampling 100,000 True Positives...")
    gt_pairs = set()
    pos_samples = []
    
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for r in reader:
            if len(r) >= 2:
                s1 = r[0].strip()
                targets = [t.strip() for t in r[1].split(',') if t.strip()]
                for t in targets:
                    gt_pairs.add((s1, t))
                    if len(pos_samples) < 100000 and s1 in s1_meta and t in target_meta:
                        pos_samples.append((s1, t))

    print(f"Sampled {len(pos_samples):,} Positives.")

    # 4. Generate 500,000 Balanced Negatives (Self-Contained, No External File Needed)
    print("[4/5] Generating 500,000 In-Memory Hard & Diverse Negatives...")
    neg_samples = []
    sampled_s1_list = [s1 for s1, _ in pos_samples]
    
    # 5:1 Negative Ratio
    while len(neg_samples) < 500000:
        s1 = random.choice(sampled_s1_list)
        t = random.choice(target_ids)
        if (s1, t) not in gt_pairs:
            neg_samples.append((s1, t))

    print(f"Generated {len(neg_samples):,} Negatives (1:5 Pos-to-Neg Ratio).")

    # Vectorize Features
    print("Vectorizing 9-Dimensional Feature Tensors...")
    X = []
    y = []

    for s1, t in pos_samples:
        s1_n, s1_a = s1_meta[s1]
        t_n, t_a = target_meta[t]
        X.append(extract_9_features(s1_n, s1_a, t_n, t_a, idf_dict))
        y.append(1)

    for s1, t in neg_samples:
        s1_n, s1_a = s1_meta[s1]
        t_n, t_a = target_meta[t]
        X.append(extract_9_features(s1_n, s1_a, t_n, t_a, idf_dict))
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int8)

    # 5. Train LightGBM
    print("[5/5] Training LightGBM Monotonic GBDT (4x FP Penalty)...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'scale_pos_weight': 0.25, # 4x penalty on FP
        'monotone_constraints': [1, 1, 1, 1, 1, 1, 1, -1, 0],
        'feature_fraction': 0.85,
        'bagging_fraction': 0.80,
        'bagging_freq': 5,
        'n_estimators': 350,
        'verbose': -1,
        'random_state': 42
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=True)]
    )

    model.save_model(MODEL_PATH)
    total_time = time.time() - t0
    print("\n" + "=" * 75)
    print(f"✅ MODEL ARTIFACT CREATED: {MODEL_PATH}")
    print(f"Runtime: {total_time:.1f}s ({total_time/60:.2f} mins)")
    print("=" * 75)

if __name__ == '__main__':
    main()