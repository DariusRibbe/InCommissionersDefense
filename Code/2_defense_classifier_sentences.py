
from transformers import pipeline
import pandas as pd
from tqdm.auto import tqdm
import torch
import os

# PATHS & SETTINGS


INPUT_CSV = "/fs/dss/work/role6027/df_sentences_defence_classified.csv"
OUT_DIR   = "/fs/dss/work/role6027/defence_bertopic_network_out"

TEXT_COLUMN = "satz_text"   # sentence column
CHUNKSIZE = 2000
BATCH_SIZE = 16
THRESHOLD = 0.7
DEVICE = 0 if torch.cuda.is_available() else -1

# Santiy checks 

if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input not found: {INPUT_CSV}")

# optional: verify column exists
tmp = pd.read_csv(INPUT_CSV, nrows=1)
if TEXT_COLUMN not in tmp.columns:
    raise ValueError(f"Column '{TEXT_COLUMN}' not found. Columns are: {list(tmp.columns)}")

# ZERO-SHOT PIPELINE

classifier = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    device=DEVICE
)

LABELS = [
    "defence (general): military, defence or security policy issues in a European or EU context",

    "building up European military capabilities and readiness to deter armed aggression and defend the EU and its Member States",

    "strengthening the European defence industrial base, armaments production, or joint procurement of weapons and military equipment in the EU",

    "collective defence of Europe together with allies such as NATO, including deterrence, military mobility, or transatlantic security",

    "supporting Ukraine as part of European defence against Russian aggression",

    "protecting critical infrastructure or strategic domains such as space, cyber space or undersea cables as part of EU security and defence",

    "achieving European strategic autonomy in security and defence so that Europe can defend and protect itself without over-reliance on others",

    "other EU policies that are not related to defence or security",
]

HYPOTHESIS_TEMPLATE = "This sentence is about {}."


# BATCH CLASSIFICATION


def classify_batch(sentences):
    results = classifier(
        sentences,
        LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        multi_label=True,
        truncation=True,
        batch_size=BATCH_SIZE
    )

    rows = []

    for res in results:
        scores = dict(zip(res["labels"], res["scores"]))

        row = {
            # individual defence dimensions
            "defence_general": scores.get(LABELS[0], 0.0),
            "defence_deterrence_readiness": scores.get(LABELS[1], 0.0),
            "defence_industry": scores.get(LABELS[2], 0.0),
            "defence_collective_nato": scores.get(LABELS[3], 0.0),
            "defence_ukraine": scores.get(LABELS[4], 0.0),
            "defence_strategic_domains": scores.get(LABELS[5], 0.0),
            "defence_strategic_autonomy": scores.get(LABELS[6], 0.0),
            "defence_non_defence_other": scores.get(LABELS[7], 0.0),
        }

        # overall defence score (max over defence-related categories)
        row["defence_score_overall"] = max(
            row["defence_general"],
            row["defence_deterrence_readiness"],
            row["defence_industry"],
            row["defence_collective_nato"],
            row["defence_ukraine"],
            row["defence_strategic_domains"],
            row["defence_strategic_autonomy"],
        )

        # binary indicator
        row["is_european_defence"] = row["defence_score_overall"] >= THRESHOLD

        rows.append(row)

    return pd.DataFrame(rows)

# RUN ON CSV (STREAMING)

def run():
    first_chunk = True

    for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNKSIZE),
                      desc="Classifying sentences"):

        sentences = chunk[TEXT_COLUMN].astype(str).fillna("").tolist()

        scores_df = classify_batch(sentences)

        out = pd.concat([chunk.reset_index(drop=True), scores_df], axis=1)

        out.to_csv(
            OUTPUT_CSV,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8"
        )

        first_chunk = False

    print("Finished. Output saved to:")
    print(OUTPUT_CSV)

# 5) EXECUTE

if __name__ == "__main__":
    run()
