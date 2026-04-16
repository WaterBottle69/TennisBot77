import pandas as pd
import numpy as np
import os
import glob
import logging
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

log = logging.getLogger(__name__)

def build_sequential_dataset(data_dir, sequence_length=10):
    """
    Parses the massive monolithic CSV files containing historical tennis matches (1998-2024),
    converting them into overlapping tensor sequences of length `sequence_length` for PyTorch's LSTM.
    """
    log.info(f"Loading datasets from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, 'atp_matches_*.csv'))
    
    df_list = [pd.read_csv(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
        
    # Sort chronologically by tourney_date to absolutely prevent lookahead leakage
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['tourney_date', 'winner_id', 'loser_id']).sort_values('tourney_date')
    
    player_histories = defaultdict(list)
    
    # Pre-allocate mappings
    surface_map = {'Hard': 1, 'Clay': 2, 'Grass': 3}
    num_players = max(df['winner_id'].max(), df['loser_id'].max())
    
    X_pA_id, X_pB_id = [], []
    X_seq_A, X_seq_B = [], []
    X_seqA_len, X_seqB_len = [], []
    X_surf, X_tourn = [], []
    X_match_feats = []
    Y_tgt = []
    
    log.info("Iterating through chronological match timeline to build sequential embeddings...")
    for idx, row in df.iterrows():
        wid, lid = int(row['winner_id']), int(row['loser_id'])
        
        # Build exact Match level features
        surf = surface_map.get(row['surface'], 0)
        # simplistic tournament mapping (GrandSlam=1, Masters=2, default=0)
        tourn = 1 if 'Grand Slam' in str(row.get('tourney_level', '')) else 0
        
        # Retrieve history
        hist_W = player_histories[wid][-sequence_length:]
        hist_L = player_histories[lid][-sequence_length:]
        
        # We need N matches minimum or padding (omitted for brevity, handled in PyTorch DataLoader padding usually)
        # A simple feature slice: [opp_elo, surf, result, days_since]
        # To avoid massive file complexity in python generator, assume we pad short histories with 0 vectors
        def pad_sequence(hist, target_len=10):
            seq = np.zeros((target_len, 4))
            for i, h in enumerate(hist):
                seq[i] = h
            return seq.tolist(), len(hist)
            
        seqW, lenW = pad_sequence(hist_W, target_len=sequence_length)
        seqL, lenL = pad_sequence(hist_L, target_len=sequence_length)
        
        # Construct symmetric dataset randomly assigning winner to A or B
        if np.random.rand() > 0.5:
            # Winner is A
            X_pA_id.append(wid)
            X_pB_id.append(lid)
            X_seq_A.append(seqW)
            X_seq_B.append(seqL)
            X_seqA_len.append(lenW)
            X_seqB_len.append(lenL)
            Y_tgt.append(1.0)
            
            pA_rank = float(row.get('winner_rank', 50))
            pB_rank = float(row.get('loser_rank', 50))
        else:
            # Winner is B
            X_pA_id.append(lid)
            X_pB_id.append(wid)
            X_seq_A.append(seqL)
            X_seq_B.append(seqW)
            X_seqA_len.append(lenL)
            X_seqB_len.append(lenW)
            Y_tgt.append(0.0)
            
            pA_rank = float(row.get('loser_rank', 50))
            pB_rank = float(row.get('winner_rank', 50))
            
        X_surf.append(surf)
        X_tourn.append(tourn)
        
        # Match Features (Rank gaps etc)
        X_match_feats.append([
            pA_rank - pB_rank,
            pA_rank,
            pB_rank
            # Other features like height, serve_pct, etc.
        ])
        
        # Update Timeline (Push this match to both players' history queues for FUTURE predictions)
        date_num = row['tourney_date'].timestamp()
        
        # Example push to winner: [opp_elo (approx), surf, 1 (Win), dates_since]
        player_histories[wid].append([pB_rank, surf, 1.0, 0.0]) # 0.0 date gap placeholder
        player_histories[lid].append([pA_rank, surf, 0.0, 0.0])

    log.info(f"Built sequential dataset with {len(Y_tgt)} timelines.")
    
    # Scale match features
    scaler = StandardScaler()
    X_match_feats = scaler.fit_transform(X_match_feats)
    
    return {
        'pA_id': np.array(X_pA_id), 'pB_id': np.array(X_pB_id),
        'seq_A': np.array(X_seq_A), 'seq_B': np.array(X_seq_B),
        'seqA_len': np.array(X_seqA_len), 'seqB_len': np.array(X_seqB_len),
        'surface': np.array(X_surf), 'tourney': np.array(X_tourn),
        'match_feats': X_match_feats,
        'targets': np.array(Y_tgt)
    }

if __name__ == "__main__":
    import joblib
    # Target directory instead of just clean file
    data_path = '../../tennis_atp-master'
    
    log.info(f"Starting dataset build from raw ATP files in {data_path}...")
    dataset = build_sequential_dataset(data_path, sequence_length=10)
    
    if dataset is not None:
        save_path = 'pytorch_tennis_dataset.joblib'
        joblib.dump(dataset, save_path)
        log.info(f"Successfully saved PyTorch dataset to {save_path}!")
        log.info(f"Dataset Size: {len(dataset['targets'])} matches generated.")

