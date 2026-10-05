import csv
import os
import re
import time
from collections import defaultdict
import lightgbm as lgb
import numpy as np

OUTPUT_DIR = "output"
TEST_DIR = os.path.join("dataset", "test")

S1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
S2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
S3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")
CAND_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs_v8_final.tsv")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission_v8.csv")

# Model path resolution
m_path_txt = os.path.join(OUTPUT_DIR, "lgbm_v8_model.txt")
m_path_raw = os.path.join(OUTPUT_DIR, "lgbm_v8_model")
MODEL_PATH = m_path_txt if os.path.exists(m_path_txt) else m_path_raw

STOPWORDS = {
    "pvt",
    "ltd",
    "limited",
    "private",
    "enterprises",
    "corp",
    "co",
    "llp",
    "inc",
}


def clean_toks(text):
  if not text:
    return set()
  clean = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
  return {t for t in clean.split() if t not in STOPWORDS and len(t) > 2}


def extract_pin(text):
  if not text:
    return ""
  m = re.search(r"\b[1-9][0-9]{5}\b", str(text))
  return m.group(0) if m else ""


print(f"Loading LightGBM model from {MODEL_PATH}...")
model = lgb.Booster(model_file=MODEL_PATH)

print("Loading Target Lookup Maps...")
target_data = {}
for path, code in [(S2_PATH, 2), (S3_PATH, 3)]:
  if os.path.exists(path):
    prefix = "s2" if code == 2 else "s3"
    with open(path, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f, delimiter="\t")
      for r in reader:
        eid = (
            r.get(f"{prefix}_entity_id") or r.get("entity_id") or ""
        ).strip()
        if eid:
          target_data[eid] = (
              clean_toks(r.get("business_name", "")),
              extract_pin(r.get("business_address", "")),
              code,
          )

print("Loading S1 Lookup Maps...")
s1_data = {}
with open(S1_PATH, "r", encoding="utf-8") as f:
  reader = csv.DictReader(f, delimiter="\t")
  for r in reader:
    eid = (r.get("source1_entity_id") or r.get("entity_id") or "").strip()
    if eid:
      s1_data[eid] = (
          clean_toks(r.get("business_name", "")),
          extract_pin(r.get("business_address", "")),
      )

print("Directly scoring candidate pairs...")
t0 = time.time()
total_matches = 0

with open(CAND_PATH, "r", encoding="utf-8") as f_cand, open(
    SUBMISSION_PATH, "w", encoding="utf-8", newline=""
) as f_out:

  reader = csv.reader(f_cand, delimiter="\t")
  writer = csv.writer(f_out)
  writer.writerow(["source1_entity_id", "matched_entity_ids"])

  first = next(reader, None)
  if first and (first[0].startswith("source1") or first[0].startswith("s1")):
    pass
  elif first:
    f_cand.seek(0)

  for idx, row in enumerate(reader):
    if len(row) < 2:
      continue
    s1_id = row[0].strip()
    raw_targets = [t.strip() for t in row[1].split(",") if t.strip()]

    if s1_id not in s1_data or not raw_targets:
      writer.writerow([s1_id, ""])
      continue

    s1_toks, s1_pin = s1_data[s1_id]
    features = []
    cands = []

    for tid in raw_targets:
      if tid not in target_data:
        continue
      t_toks, t_pin, src_code = target_data[tid]

      inter = s1_toks.intersection(t_toks)
      union = s1_toks.union(t_toks)
      f1 = len(inter) / len(union) if union else 0.0
      f2 = f1
      f3 = (
          (2.0 * len(inter)) / (len(s1_toks) + len(t_toks))
          if (s1_toks and t_toks)
          else 0.0
      )
      f4 = (
          min(len(s1_toks), len(t_toks)) / max(len(s1_toks), len(t_toks))
          if (s1_toks and t_toks)
          else 0.0
      )
      f5 = 5.0 if inter else 0.0
      f6 = float(len(inter) * 3.0)
      f7 = 1.0 if (s1_pin and t_pin and s1_pin == t_pin) else 0.0
      f8 = 1.0 if (s1_pin and t_pin and s1_pin != t_pin) else 0.0
      f9 = 1.0 if (not s1_pin or not t_pin) else 0.0

      features.append([f1, f2, f3, f4, f5, f6, f7, f8, f9])
      cands.append((tid, src_code))

    if not features:
      writer.writerow([s1_id, ""])
      continue

    scores = model.predict(np.array(features, dtype=np.float32))

    best_s2 = (None, -1.0)
    best_s3 = (None, -1.0)

    for (tid, src_code), score in zip(cands, scores):
      if score >= 0.50:
        if src_code == 2 and score > best_s2[1]:
          best_s2 = (tid, score)
        elif src_code == 3 and score > best_s3[1]:
          best_s3 = (tid, score)

    selected = []
    if best_s2[0]:
      selected.append(best_s2[0])
    if best_s3[0]:
      selected.append(best_s3[0])

    writer.writerow([s1_id, ",".join(selected)])
    total_matches += len(selected)

    if (idx + 1) % 100000 == 0:
      print(f" -> Processed {idx + 1:,} S1 rows | Matches: {total_matches:,}")

print(f"Completed in {time.time() - t0:.1f}s. File written: {SUBMISSION_PATH}")