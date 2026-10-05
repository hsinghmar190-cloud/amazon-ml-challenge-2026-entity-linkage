import os
import re
import pandas as pd
from collections import defaultdict

CURRENT_FILE = os.path.abspath(__file__)
NOTEBOOKS_DIR = os.path.dirname(CURRENT_FILE)
BASE_DIR = os.path.dirname(NOTEBOOKS_DIR)
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

print("--- Fast Vectorized Multi-Block Engine ---")
# 10k S1 and 100k S2 sample
s1_sample = pd.read_csv(os.path.join(TRAIN_DIR, "train_source1.tsv"), sep="\t", nrows=10000)
s2_sample = pd.read_csv(os.path.join(TRAIN_DIR, "train_source2.tsv"), sep="\t", nrows=100000)
gt = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t")

# Fast ground truth map
gt_dict = {}
for row in gt.itertuples(index=False):
    if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip() != "":
        matches = [m.strip() for m in str(row.matched_entity_ids).split(",")]
        s2_matches = [m for m in matches if m.startswith("S2-")]
        if s2_matches:
            gt_dict[row.source1_entity_id] = set(s2_matches)

STOP_SUFFIXES = {
    'inc', 'llc', 'ltd', 'pvt', 'co', 'corp', 'corporation', 
    'limited', 'services', 'enterprises', 'company', 'the'
}

def get_tokens(text):
    if not isinstance(text, str):
        return []
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [w for w in cleaned.split() if w not in STOP_SUFFIXES and len(w) > 1]

token_index = defaultdict(list)
two_token_index = defaultdict(list)

print("Indexing 100k records (Fast Mode)...")
# itertuples is 50x faster than iterrows
for row in s2_sample.itertuples(index=False):
    cid = row.entity_id
    country = str(row.country).strip().upper()
    tokens = get_tokens(row.business_name)
    
    if tokens:
        token_index[(country, tokens[0])].append(cid)
        longest = max(tokens, key=len)
        if len(longest) >= 4:
            token_index[(country, longest)].append(cid)
        if len(tokens) >= 2:
            bi_key = (country, tokens[0][:3] + tokens[1][:3])
            two_token_index[bi_key].append(cid)

print(f"Indexing done! Total token keys: {len(token_index):,}")

print("Evaluating Recall on Ground Truth...")
s2_ids_set = set(s2_sample['entity_id'])
evaluated_s1 = 0
recalled_s1 = 0

for row in s1_sample.itertuples(index=False):
    s1_id = row.entity_id
    if s1_id not in gt_dict:
        continue
    
    available_truth = gt_dict[s1_id].intersection(s2_ids_set)
    if not available_truth:
        continue
    
    evaluated_s1 += 1
    country = str(row.country).strip().upper()
    tokens = get_tokens(row.business_name)
    
    candidates = set()
    if tokens:
        candidates.update(token_index.get((country, tokens[0]), [])[:30])
        longest = max(tokens, key=len)
        if len(longest) >= 4:
            candidates.update(token_index.get((country, longest), [])[:30])
        if len(tokens) >= 2:
            bi_key = (country, tokens[0][:3] + tokens[1][:3])
            candidates.update(two_token_index.get(bi_key, [])[:30])
            
    if candidates.intersection(available_truth):
        recalled_s1 += 1

recall = (recalled_s1 / evaluated_s1) * 100
print(f"Evaluated S1 Entities: {evaluated_s1}")
print(f"Candidates Recalled: {recalled_s1}")
print(f"Upgraded Blocking Recall: {recall:.2f}%")