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
    player_fatigue_log = defaultdict(list)  # pid -> [(timestamp, minutes)]
    player_serve_stats = defaultdict(lambda: {'ace': 0.0, 'svpt': 0.0}) # pid -> cumulative stats
    player_clutch_stats = defaultdict(lambda: {'saved': 0.0, 'faced': 0.0})
    player_lefty_stats = defaultdict(lambda: {'wins': 0.0, 'matches': 0.0})
    player_serve_variance = defaultdict(list) # pid -> list of recent 1stIn/svpt ratios
    
    # Altitude Mapping (meters above sea level)
    altitude_map = {
        'bogota': 2640.0,
        'gstaad': 1050.0,
        'kitzbuhel': 762.0,
        'madrid': 667.0,
        'quito': 2850.0,
        'indian wells': 35.0,
        'miami': 2.0,
        'monte carlo': 50.0,
        'rome': 20.0,
        'roland garros': 35.0,
        'wimbledon': 30.0,
        'us open': 5.0,
        'cincinnati': 147.0,
        'paris': 35.0,
        'shanghai': 4.0,
        'australian open': 5.0
    }
    
    # Historical Average Weather Map: 'tourney_name': (Temp C, Humidity %)
    weather_map = {
        'australian open': (26.0, 50.0),
        'indian wells': (25.0, 20.0),
        'miami': (24.0, 65.0),
        'monte carlo': (15.0, 70.0),
        'madrid': (20.0, 45.0),
        'rome': (21.0, 60.0),
        'roland garros': (20.0, 65.0),
        'wimbledon': (21.0, 60.0),
        'us open': (26.0, 60.0),
        'cincinnati': (25.0, 65.0),
        'paris': (12.0, 80.0),
        'shanghai': (22.0, 65.0),
        'bogota': (18.0, 75.0),
        'gstaad': (20.0, 50.0),
    }
    
    # CPI Hardcoded Mapping
    cpi_map = {
        'Indian Wells': 27, 'Miami': 35, 'Monte Carlo': 22, 'Madrid': 33, 'Rome': 24,
        'Roland Garros': 21, 'Wimbledon': 37, 'US Open': 43, 'Cincinnati': 43,
        'Paris': 39, 'Shanghai': 44, 'Australian Open': 40
    }
    
    # Pre-allocate mappings
    surface_map = {'Hard': 1, 'Clay': 2, 'Grass': 3}
    num_players = max(df['winner_id'].max(), df['loser_id'].max())
    
    # Fill NAs to avoid float conversion errors
    df['winner_rank'] = df['winner_rank'].fillna(50)
    df['loser_rank'] = df['loser_rank'].fillna(50)
    df['winner_age'] = df['winner_age'].fillna(25)
    df['loser_age'] = df['loser_age'].fillna(25)
    df['winner_ht'] = df['winner_ht'].fillna(185)
    df['loser_ht'] = df['loser_ht'].fillna(185)
    df['best_of'] = df['best_of'].fillna(3)

    # Frequency filter for embeddings (prevent overfitting on rare players)
    player_counts = defaultdict(int)
    for _, row in df.iterrows():
        player_counts[int(row['winner_id'])] += 1
        player_counts[int(row['loser_id'])] += 1
    
    def map_id(pid):
        return pid if player_counts[pid] >= 20 else 0
    
    X_pA_id, X_pB_id = [], []
    X_seq_A, X_seq_B = [], []
    X_seqA_len, X_seqB_len = [], []
    X_surf, X_tourn = [], []
    X_match_feats = []
    Y_tgt = []
    
    log.info("Iterating through chronological match timeline to build sequential embeddings...")
    # FIX: Seed the A/B assignment RNG so dataset builds are deterministic.
    # Without this, each run produces a different training dataset, making
    # experiment results irreproducible and train/val splits inconsistent.
    np.random.seed(42)
    for idx, row in df.iterrows():
        wid_raw, lid_raw = int(row['winner_id']), int(row['loser_id'])
        wid = map_id(wid_raw)
        lid = map_id(lid_raw)
        
        # Build exact Match level features
        surf = surface_map.get(row['surface'], 0)
        # simplistic tournament mapping (GrandSlam=1, Masters=2, default=0)
        tourn = 1 if 'Grand Slam' in str(row.get('tourney_level', '')) else 0
        
        # Calculate CPI
        tourney_name = str(row.get('tourney_name', ''))
        cpi = 35.0  # default medium
        for t_name, t_cpi in cpi_map.items():
            if t_name in tourney_name:
                cpi = float(t_cpi)
                break
                
        date_num = row['tourney_date'].timestamp()
        
        # Calculate Fatigue (Trailing 7-day minutes)
        def get_fatigue(pid, current_time):
            player_fatigue_log[pid] = [m for m in player_fatigue_log[pid] if current_time - m[0] <= 7 * 86400]
            return sum(m[1] for m in player_fatigue_log[pid])
            
        w_fatigue = get_fatigue(wid_raw, date_num)
        l_fatigue = get_fatigue(lid_raw, date_num)
        
        # Calculate Stylistic Archetypes (Servebot index) using trailing HISTORY ONLY
        def get_historical_archetype(pid):
            stats = player_serve_stats[pid]
            ace = stats['ace']
            svpt = stats['svpt']
            if svpt == 0: 
                return 0.5  # Default if no history
            return min(1.0, (ace / svpt) * 5.0)  # ~20% ace rate caps at 1.0
            
        w_arch = get_historical_archetype(wid_raw)
        l_arch = get_historical_archetype(lid_raw)
        
        # Calculate Altitude & Weather Physics
        tname = str(row.get('tourney_name', '')).lower()
        altitude = 0.0
        temp_c = 22.0     # Default neutral temp
        humidity = 50.0   # Default neutral humidity
        
        for k, v in altitude_map.items():
            if k in tname:
                altitude = v
                break
                
        for k, v in weather_map.items():
            if k in tname:
                temp_c, humidity = v
                break
                
        # Air Density Calculation (approximation multiplier)
        # Drops with altitude, drops with heat, drops slightly with humidity
        # Base density at 0m, 15C, 0% hum = ~1.225 kg/m^3. We normalize to 1.0.
        air_density = (1.0 - (altitude / 10400.0)) * (288.15 / (273.15 + temp_c)) * (1.0 - (humidity * 0.0005))
        
        
        # Calculate Clutch Factor
        def get_clutch(pid):
            s = player_clutch_stats[pid]
            if s['faced'] == 0: return 0.5
            return s['saved'] / s['faced']
            
        w_clutch = get_clutch(wid_raw)
        l_clutch = get_clutch(lid_raw)
        
        # Calculate Lefty Win Rate
        def get_lefty_winrate(pid):
            s = player_lefty_stats[pid]
            if s['matches'] == 0: return 0.5
            return s['wins'] / s['matches']
            
        w_lefty_winrate = get_lefty_winrate(wid_raw)
        l_lefty_winrate = get_lefty_winrate(lid_raw)
        
        # Calculate 1st Serve Variance (Rolling 5-match StdDev)
        def get_serve_var(pid):
            history = player_serve_variance[pid][-5:]
            if len(history) < 2: return 0.0
            return np.std(history)
            
        w_serve_var = get_serve_var(wid_raw)
        l_serve_var = get_serve_var(lid_raw)
        
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
            
            pA_rank = float(row['winner_rank'])
            pB_rank = float(row['loser_rank'])
            pA_age = float(row['winner_age'])
            pB_age = float(row['loser_age'])
            pA_ht = float(row['winner_ht'])
            pB_ht = float(row['loser_ht'])
            pA_hand = 1.0 if row.get('winner_hand') == 'R' else 0.0
            pB_hand = 1.0 if row.get('loser_hand') == 'R' else 0.0
            
            pA_fatigue, pB_fatigue = w_fatigue, l_fatigue
            pA_arch, pB_arch = w_arch, l_arch
            pA_clutch, pB_clutch = w_clutch, l_clutch
            pA_lefty_winrate, pB_lefty_winrate = w_lefty_winrate, l_lefty_winrate
            pA_serve_var, pB_serve_var = w_serve_var, l_serve_var
        else:
            # Winner is B
            X_pA_id.append(lid)
            X_pB_id.append(wid)
            X_seq_A.append(seqL)
            X_seq_B.append(seqW)
            X_seqA_len.append(lenL)
            X_seqB_len.append(lenW)
            Y_tgt.append(0.0)
            
            pA_rank = float(row['loser_rank'])
            pB_rank = float(row['winner_rank'])
            pA_age = float(row['loser_age'])
            pB_age = float(row['winner_age'])
            pA_ht = float(row['loser_ht'])
            pB_ht = float(row['winner_ht'])
            pA_hand = 1.0 if row.get('loser_hand') == 'R' else 0.0
            pB_hand = 1.0 if row.get('winner_hand') == 'R' else 0.0
            
            pA_fatigue, pB_fatigue = l_fatigue, w_fatigue
            pA_arch, pB_arch = l_arch, w_arch
            pA_clutch, pB_clutch = l_clutch, w_clutch
            pA_lefty_winrate, pB_lefty_winrate = l_lefty_winrate, w_lefty_winrate
            pA_serve_var, pB_serve_var = l_serve_var, w_serve_var
            
        X_surf.append(surf)
        X_tourn.append(tourn)
        
        # Match Features exactly matching ml_engine.py
        surf_h = 1.0 if surf == 1 else 0.0
        surf_c = 1.0 if surf == 2 else 0.0
        surf_g = 1.0 if surf == 3 else 0.0
        best_of = float(row['best_of'])
        elo1, elo2 = 1000.0, 1000.0  # Fallback since Elo not in raw CSV
        
        X_match_feats.append([
            pA_rank - pB_rank,
            pA_rank,
            pB_rank,
            elo1, elo2,
            pA_age, pB_age,
            pA_ht, pB_ht,
            pA_hand, pB_hand,
            best_of,
            surf_h, surf_c, surf_g,
            cpi,
            pA_fatigue, pB_fatigue,
            pA_arch, pB_arch,
            pA_clutch, pB_clutch,
            pA_lefty_winrate, pB_lefty_winrate,
            pA_serve_var, pB_serve_var,
            altitude,
            air_density
        ])
        
        # Update Timeline (Push this match to both players' history queues for FUTURE predictions)
        # Update Fatigue Log
        minutes = float(row.get('minutes') or 90.0)
        player_fatigue_log[wid_raw].append((date_num, minutes))
        player_fatigue_log[lid_raw].append((date_num, minutes))
        
        # Update Archetype Log
        player_serve_stats[wid_raw]['ace'] += float(row.get('w_ace') or 0.0)
        player_serve_stats[wid_raw]['svpt'] += float(row.get('w_svpt') or 0.0)
        player_serve_stats[lid_raw]['ace'] += float(row.get('l_ace') or 0.0)
        player_serve_stats[lid_raw]['svpt'] += float(row.get('l_svpt') or 0.0)
        
        # Update Clutch Stats
        player_clutch_stats[wid_raw]['saved'] += float(row.get('w_bpSaved') or 0.0)
        player_clutch_stats[wid_raw]['faced'] += float(row.get('w_bpFaced') or 0.0)
        player_clutch_stats[lid_raw]['saved'] += float(row.get('l_bpSaved') or 0.0)
        player_clutch_stats[lid_raw]['faced'] += float(row.get('l_bpFaced') or 0.0)
        
        # Update Lefty Stats
        w_hand = row.get('winner_hand')
        l_hand = row.get('loser_hand')
        if l_hand == 'L':
            player_lefty_stats[wid_raw]['matches'] += 1
            player_lefty_stats[wid_raw]['wins'] += 1
        if w_hand == 'L':
            player_lefty_stats[lid_raw]['matches'] += 1
            # loser lost, so wins += 0
            
        # Update Serve Variance
        w_svpt = float(row.get('w_svpt') or 1.0)
        if w_svpt > 0:
            player_serve_variance[wid_raw].append(float(row.get('w_1stIn') or 0.0) / w_svpt)
        l_svpt = float(row.get('l_svpt') or 1.0)
        if l_svpt > 0:
            player_serve_variance[lid_raw].append(float(row.get('l_1stIn') or 0.0) / l_svpt)
        
        # Example push to winner: [opp_elo (approx), surf, 1 (Win), dates_since]
        player_histories[wid].append([pB_rank, surf, 1.0, 0.0]) # 0.0 date gap placeholder
        player_histories[lid].append([pA_rank, surf, 0.0, 0.0])

    log.info(f"Built sequential dataset with {len(Y_tgt)} timelines.")
    
    # Scale match features and SAVE the scaler for live inference!
    scaler = StandardScaler()
    X_match_feats = scaler.fit_transform(X_match_feats)
    joblib.dump(scaler, 'nn_scaler.joblib')
    log.info("Saved feature scaler to nn_scaler.joblib")
    
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
    data_path = 'massive_tennis_dataset/atp_tour'
    
    log.info(f"Starting dataset build from raw ATP files in {data_path}...")
    dataset = build_sequential_dataset(data_path, sequence_length=10)
    
    if dataset is not None:
        save_path = 'pytorch_tennis_dataset.joblib'
        joblib.dump(dataset, save_path)
        log.info(f"Successfully saved PyTorch dataset to {save_path}!")
        log.info(f"Dataset Size: {len(dataset['targets'])} matches generated.")

