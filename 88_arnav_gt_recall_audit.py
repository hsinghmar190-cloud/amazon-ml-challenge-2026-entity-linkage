import pandas as pd
import os
import csv

# ==============================================================================
# PATHS
# ==============================================================================
GT_PATH = os.path.join("dataset", "train", "train_ground_truth.tsv")
CAND_PATH = os.path.join("output", "v8_train_candidate_pairs.tsv")
S1_PATH = os.path.join("dataset", "train", "train_source1.tsv")
S2_PATH = os.path.join("dataset", "train", "train_source2.tsv")
S3_PATH = os.path.join("dataset", "train", "train_source3.tsv")

def main():
    print("="*75)
    print("--- [ARNAV AUDIT: UPSTREAM GT RECALL EVALUATOR] ---")
    print("="*75)

    if not os.path.exists(CAND_PATH):
        print(f"ERROR: Dheeraj's candidate file missing at {CAND_PATH}")
        return

    # 1. Load addresses to identify missing-address GT pairs
    print("[1/3] Loading addresses for sparsity mapping...")
    s1_addr = pd.read_csv(S1_PATH, sep='\t', usecols=['entity_id', 'business_address']).set_index('entity_id')['business_address'].fillna("").to_dict()
    s2_addr = pd.read_csv(S2_PATH, sep='\t', usecols=['entity_id', 'business_address']).set_index('entity_id')['business_address'].fillna("").to_dict()
    s3_addr = pd.read_csv(S3_PATH, sep='\t', usecols=['entity_id', 'business_address']).set_index('entity_id')['business_address'].fillna("").to_dict()
    
    target_addr = {**s2_addr, **s3_addr}

    # 2. Parse Ground Truth
    print("[2/3] Parsing Ground Truth & isolating missing-address pairs...")
    total_gt = 0
    missing_addr_gt = 0
    gt_pairs = set()
    missing_gt_pairs = set()

    with open(GT_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 2: continue
            s1 = row[0].strip()
            targets = [t.strip() for t in row[1].split(",") if t.strip()]
            
            s1_a = str(s1_addr.get(s1, "")).strip()
            
            for t in targets:
                gt_pairs.add((s1, t))
                total_gt += 1
                
                t_a = str(target_addr.get(t, "")).strip()
                if not s1_a or not t_a or s1_a.lower() == 'nan' or t_a.lower() == 'nan':
                    missing_gt_pairs.add((s1, t))
                    missing_addr_gt += 1

    # 3. Stream Candidate Pairs and Calculate Recall
    print("[3/3] Evaluating Dheeraj's Candidate Pool against GT...")
    found_total = 0
    found_missing = 0

    with open(CAND_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        
        # Check if first row is header
        first_row = next(reader, None)
        if first_row and first_row[0] != "source1_entity_id":
            # Process it if it's actual data
            s1 = first_row[0].strip()
            cands = [c.strip() for c in first_row[1].split(",") if c.strip()] if len(first_row) > 1 else []
            for c in cands:
                if (s1, c) in gt_pairs: found_total += 1
                if (s1, c) in missing_gt_pairs: found_missing += 1

        # Process rest of file
        for row in reader:
            if len(row) < 2: continue
            s1 = row[0].strip()
            cands = [c.strip() for c in row[1].split(",") if c.strip()]
            
            for c in cands:
                if (s1, c) in gt_pairs:
                    found_total += 1
                if (s1, c) in missing_gt_pairs:
                    found_missing += 1

    print("\n" + "="*75)
    print("--- [ARNAV AUDIT RESULTS] ---")
    print(f"Overall GT Recall:         {found_total:,} / {total_gt:,} ({(found_total/total_gt)*100:.2f}%)")
    print(f"Missing-Address GT Recall: {found_missing:,} / {missing_addr_gt:,} ({(found_missing/missing_addr_gt)*100:.2f}%)")
    print("="*75)

if __name__ == "__main__":
    main()