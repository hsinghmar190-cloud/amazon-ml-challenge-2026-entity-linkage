import pandas as pd
import numpy as np
import csv
import re
import math
import random
from difflib import SequenceMatcher
import os

# ==============================================================================
# PATHS
# ==============================================================================
TRAIN_DIR = os.path.join("dataset", "train")
OUTPUT_DIR = "output"
GT_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
S1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
S2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
S3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
FEATURE_OUTPUT = os.path.join(OUTPUT_DIR, "v8_train_features.csv")

# ==============================================================================
# UTILITIES
# ==============================================================================
def clean_text(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^a-z0-9\s]', '', text.lower()).strip()

def char_ngrams(text, n=3):
    clean = re.sub(r'[^a-z0-9]', '', str(text).lower())
    if len(clean) < n: return set([clean]) if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}

def cosine_similarity(set_a, set_b):
    if not set_a or not set_b: return 0.0
    inter = len(set_a.intersection(set_b))
    norm = math.sqrt(len(set_a)) * math.sqrt(len(set_b))
    return inter / norm if norm > 0 else 0.0

def jaccard_similarity(set_a, set_b):
    if not set_a or not set_b: return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0

def main():
    print("="*75)
    print("--- [V8 PHASE 2: ML FEATURE ENGINEERING (BUILDING TENSORS)] ---")
    print("="*75)

    # 1. Load Data with Pandas (Only needed columns for memory safety)
    print("Loading datasets...")
    cols = ['entity_id', 'business_name', 'business_address']
    
    s1 = pd.read_csv(S1_PATH, sep='\t', usecols=lambda c: c in cols or c == 'source1_entity_id').fillna("")
    s1.rename(columns={'source1_entity_id': 'entity_id'}, inplace=True)
    
    s2 = pd.read_csv(S2_PATH, sep='\t', usecols=lambda c: c in cols or c == 'source2_entity_id').fillna("")
    s2.rename(columns={'source2_entity_id': 'entity_id'}, inplace=True)
    
    s3 = pd.read_csv(S3_PATH, sep='\t', usecols=lambda c: c in cols or c == 'source3_entity_id').fillna("")
    s3.rename(columns={'source3_entity_id': 'entity_id'}, inplace=True)
    
    gt = pd.read_csv(GT_PATH, sep='\t').fillna("")

    # Create fast lookup dictionaries
    print("Building lookup indexes...")
    s1_dict = s1.set_index('entity_id').to_dict('index')
    
    targets = pd.concat([s2, s3])
    targets_dict = targets.set_index('entity_id').to_dict('index')
    target_ids_list = list(targets_dict.keys())

    features = []
    
    # 2. Extract TRUE POSITIVES (Limit to 10,000 for fast training)
    print("Extracting Features for True Positives (Label 1)...")
    tp_count = 0
    gt_pairs = set()
    
    for _, row in gt.iterrows():
        if tp_count >= 10000: break
        
        s1_id = str(row.iloc[0]).strip()
        matches = [m.strip() for m in str(row.iloc[1]).split(',') if m.strip()]
        
        if s1_id not in s1_dict: continue
        
        for t_id in matches:
            if t_id in targets_dict:
                gt_pairs.add((s1_id, t_id))
                s1_data = s1_dict[s1_id]
                t_data = targets_dict[t_id]
                
                # FEATURE CALCULATION
                s1_name, t_name = clean_text(s1_data['business_name']), clean_text(t_data['business_name'])
                s1_addr, t_addr = clean_text(s1_data['business_address']), clean_text(t_data['business_address'])
                
                name_cos = cosine_similarity(char_ngrams(s1_name), char_ngrams(t_name))
                name_seq = SequenceMatcher(None, s1_name, t_name).ratio()
                
                is_addr_missing = 1 if (not s1_addr or not t_addr) else 0
                addr_cos = cosine_similarity(char_ngrams(s1_addr), char_ngrams(t_addr)) if not is_addr_missing else 0.0
                
                features.append({
                    's1_id': s1_id, 't_id': t_id,
                    'name_cosine': name_cos,
                    'name_seq_ratio': name_seq,
                    'addr_cosine': addr_cos,
                    'is_addr_missing': is_addr_missing,
                    'label': 1  # 1 = True Positive
                })
                tp_count += 1
                if tp_count >= 10000: break

    # 3. Generate HARD NEGATIVES (Label 0)
    print("Generating Features for Hard Negatives (Label 0)...")
    fp_count = 0
    
    for s1_id, s1_data in s1_dict.items():
        if fp_count >= 20000: break
        
        # Pick a random target to simulate negative
        t_id = random.choice(target_ids_list)
        if (s1_id, t_id) in gt_pairs: continue # Skip if it's actually a True Positive
        
        t_data = targets_dict[t_id]
        
        s1_name, t_name = clean_text(s1_data['business_name']), clean_text(t_data['business_name'])
        s1_addr, t_addr = clean_text(s1_data['business_address']), clean_text(t_data['business_address'])
        
        name_cos = cosine_similarity(char_ngrams(s1_name), char_ngrams(t_name))
        name_seq = SequenceMatcher(None, s1_name, t_name).ratio()
        
        is_addr_missing = 1 if (not s1_addr or not t_addr) else 0
        addr_cos = cosine_similarity(char_ngrams(s1_addr), char_ngrams(t_addr)) if not is_addr_missing else 0.0
        
        features.append({
            's1_id': s1_id, 't_id': t_id,
            'name_cosine': name_cos,
            'name_seq_ratio': name_seq,
            'addr_cosine': addr_cos,
            'is_addr_missing': is_addr_missing,
            'label': 0  # 0 = False Positive
        })
        fp_count += 1

    # 4. Save to CSV
    print("\nSaving Training Features to CSV...")
    df = pd.DataFrame(features)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df.to_csv(FEATURE_OUTPUT, index=False)
    
    print(f"File Saved: {FEATURE_OUTPUT}")
    print(f"Total Rows: {len(df)}")
    print(f"Features: {list(df.columns)}")
    print("="*75)
    print("Ready for LightGBM Training!")

if __name__ == "__main__":
    main()