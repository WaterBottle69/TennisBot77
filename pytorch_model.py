import os
import numpy as np
import pandas as pd
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import joblib
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# --- ARCHITECTURE ---

class TennisNet(nn.Module):
    def __init__(self, num_players, embed_dim=32, seq_input_dim=4, rnn_hidden_dim=64, match_feat_dim=15, num_surfaces=3, num_tournaments=4):
        super(TennisNet, self).__init__()
        
        # 1. Player Embeddings
        # We add +1 for a "padding" or "unknown" player ID mapping to index 0
        self.player_embedding = nn.Embedding(num_players + 1, embed_dim, padding_idx=0)
        
        # 2. Sequential Encoders (Shared weights for A and B)
        # Sequence input features per timestep: [opponent_elo, surface_encoded, win_loss, days_since]
        self.rnn = nn.LSTM(
            input_size=seq_input_dim, 
            hidden_size=rnn_hidden_dim, 
            batch_first=True, 
            dropout=0.2, 
            num_layers=1
        )
        
        # 3. Categorical Embeddings for match level
        self.surface_embedding = nn.Embedding(num_surfaces + 1, 4, padding_idx=0)
        self.tourney_embedding = nn.Embedding(num_tournaments + 1, 4, padding_idx=0)
        
        # 4. Feature Combination & Fully Connected Layers
        # Concat size: Seq_A (64) + Seq_B (64) + Embed_A (32) + Embed_B (32) + Cat_Emb (8) + Match_Feats (match_feat_dim) + Diff_Feats
        # Total approx: 64+64+32+32+8+match_feat_dim
        fc_input_dim = 200 + match_feat_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, pA_id, pB_id, seqA, seqB, seqA_lens, seqB_lens, surface_cat, tourney_cat, match_feats):
        """
        seqA, seqB shape: (batch_size, seq_len, seq_input_dim)
        """
        # Embeddings
        embedA = self.player_embedding(pA_id)
        embedB = self.player_embedding(pB_id)
        
        # Process LSTM for A
        # Using basic lstm approach (we could use pack_padded_sequence for true masking)
        outA, (hn_A, cn_A) = self.rnn(seqA)
        # Extract the last valid hidden state based on lengths (simplification: just take hn_A[-1])
        seq_repA = hn_A[-1] # shape: (batch_size, rnn_hidden_dim)
        
        # Process LSTM for B
        outB, (hn_B, cn_B) = self.rnn(seqB)
        seq_repB = hn_B[-1]
        
        # Match Categories
        surf_emb = self.surface_embedding(surface_cat).squeeze(1)
        tourn_emb = self.tourney_embedding(tourney_cat).squeeze(1)
        
        # Combine all features
        x = torch.cat([
            seq_repA, seq_repB,
            embedA, embedB,
            surf_emb, tourn_emb,
            match_feats
        ], dim=1)
        
        prob = self.fc(x)
        return prob


# --- DATASET & DATALOADER ---

class TennisDataset(Dataset):
    def __init__(self, data_dict):
        """
        data_dict contains mapping of arrays
        """
        self.pA_ids = torch.tensor(data_dict['pA_id'], dtype=torch.long)
        self.pB_ids = torch.tensor(data_dict['pB_id'], dtype=torch.long)
        self.seq_A = torch.tensor(data_dict['seq_A'], dtype=torch.float32)
        self.seq_B = torch.tensor(data_dict['seq_B'], dtype=torch.float32)
        self.seqA_lens = torch.tensor(data_dict['seqA_len'], dtype=torch.long)
        self.seqB_lens = torch.tensor(data_dict['seqB_len'], dtype=torch.long)
        self.surface = torch.tensor(data_dict['surface'], dtype=torch.long)
        self.tourney = torch.tensor(data_dict['tourney'], dtype=torch.long)
        self.match_feats = torch.tensor(data_dict['match_feats'], dtype=torch.float32)
        self.targets = torch.tensor(data_dict['targets'], dtype=torch.float32)
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.pA_ids[idx], self.pB_ids[idx],
            self.seq_A[idx], self.seq_B[idx],
            self.seqA_lens[idx], self.seqB_lens[idx],
            self.surface[idx], self.tourney[idx],
            self.match_feats[idx], self.targets[idx]
        )


# --- INTEGRATION & META-MODEL ---

class XGBoostPyTorchBlender:
    """
    Learns to mix the probabilities from the Walk-Forward XGBoost baseline
    with the PyTorch LSTM sequence model effectively using Logistic Regression calibration.
    """
    def __init__(self):
        self.blender = LogisticRegression()
        self.is_fitted = False
        
    def fit(self, xgb_probs, nn_probs, targets):
        X = np.column_stack((xgb_probs, nn_probs))
        self.blender.fit(X, targets)
        self.is_fitted = True
        log.info(f"Blender calibrated. Coefficients: XGB={self.blender.coef_[0][0]:.3f}, NN={self.blender.coef_[0][1]:.3f}")
        
    def predict_proba(self, xgb_pred, nn_pred):
        if not self.is_fitted:
            # Fallback simple weighted average
            return 0.4 * xgb_pred + 0.6 * nn_pred
        X = np.array([[xgb_pred, nn_pred]])
        return self.blender.predict_proba(X)[0][1]


# --- TRAINING INFRASTRUCTURE ---

def train_neural_network(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    log.info(f"Starting neural network training on {device}...")
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            pA, pB, sqA, sqB, sqAlen, sqBlen, surf, tourn, feats, targets = [b.to(device) for b in batch]
            
            # Prevent NaN propagation from missing ranks/statistics
            feats = torch.nan_to_num(feats, nan=0.0)
            sqA = torch.nan_to_num(sqA, nan=0.0)
            sqB = torch.nan_to_num(sqB, nan=0.0)
            
            optimizer.zero_grad()
            preds = model(pA, pB, sqA, sqB, sqAlen, sqBlen, surf.unsqueeze(1), tourn.unsqueeze(1), feats).squeeze(-1)
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targs = []
        
        with torch.no_grad():
            for batch in val_loader:
                pA, pB, sqA, sqB, sqAlen, sqBlen, surf, tourn, feats, targets = [b.to(device) for b in batch]
                
                feats = torch.nan_to_num(feats, nan=0.0)
                sqA = torch.nan_to_num(sqA, nan=0.0)
                sqB = torch.nan_to_num(sqB, nan=0.0)
                
                preds = model(pA, pB, sqA, sqB, sqAlen, sqBlen, surf.unsqueeze(1), tourn.unsqueeze(1), feats).squeeze(-1)
                loss = criterion(preds, targets)
                val_loss += loss.item() * targets.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_targs.extend(targets.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_targs, np.round(all_preds))
        val_bce = log_loss(all_targs, all_preds)
        
        log.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_tennis_nn.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log.info("Early stopping triggered.")
                break
                
    log.info("Finished training.")


if __name__ == "__main__":
    import joblib
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    data_path = 'pytorch_tennis_dataset.joblib'
    if not os.path.exists(data_path):
        log.error(f"Cannot find {data_path}. Please run dataset builder first.")
    else:
        log.info(f"Loading dataset from {data_path}...")
        data_dict = joblib.load(data_path)
        
        full_dataset = TennisDataset(data_dict)
        dataset_size = len(full_dataset)
        log.info(f"Loaded dataset with {dataset_size} samples.")
        
        # Time-based splitting directly (old data used for train, new data for val)
        # Assuming the dataset builder ordered them chronologically!
        train_size = int(0.8 * dataset_size)
        indices = list(range(dataset_size))
        
        train_dataset = Subset(full_dataset, indices[:train_size])
        val_dataset = Subset(full_dataset, indices[train_size:])
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        num_players = max(data_dict['pA_id'].max(), data_dict['pB_id'].max())
        match_feat_dim = data_dict['match_feats'].shape[1]
        
        model = TennisNet(num_players=num_players, match_feat_dim=match_feat_dim)
        
        # We limit epochs to 5 for the live environment execution check
        train_neural_network(model, train_loader, val_loader, epochs=5, lr=1e-3, device='cpu')

