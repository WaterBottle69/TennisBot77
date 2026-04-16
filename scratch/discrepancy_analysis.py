
import pandas as pd
import numpy as np
import os
import glob
import json
import joblib

ATP_DIR = os.path.expanduser("~/Downloads/tennis_atp-master")
MODEL_PATH = "best_xgb_model.json"
FEATURES_PATH = "model_features.json"

def load_data():
    files = sorted(glob.glob(os.path.join(ATP_DIR, "atp_matches_[12]*.csv")))
    frames = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        df = pd.read_csv(f, low_memory=False)
        df["year"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def analyze():
    df = load_data()
    
    # Simple accuracy check (rank-based)
    df['higher_rank_won'] = (df['winner_rank'] < df['loser_rank']).astype(float)
    
    # Filter years 2022, 2023, 2024, 2026
    years = [2022, 2023, 2024, 2026]
    results = []
    for yr in years:
        yr_df = df[df['year'] == yr]
        count = len(yr_df)
        rank_acc = yr_df['higher_rank_won'].mean()
        
        # Rank diff mean
        rank_diff = (yr_df['loser_rank'] - yr_df['winner_rank']).abs().mean()
        
        results.append({
            "year": yr,
            "count": count,
            "rank_acc": rank_acc,
            "avg_rank_diff": rank_diff
        })
    
    print(pd.DataFrame(results))

    # Check the 2026 data specifically
    df2026 = df[df['year'] == 2026]
    print("\n2026 Top matches:")
    print(df2026[['winner_name', 'winner_rank', 'loser_name', 'loser_rank', 'score']].head(10))

if __name__ == "__main__":
    analyze()
