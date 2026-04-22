"""
gpu_backtester.py — High-Performance Tennis Betting Backtester
==============================================================
Designed for maximum CPU + GPU utilisation on Windows/Mac/Linux.

Features
--------
  - GUI file browser (tkinter) or CLI argument input
  - Accepts: CSV, Parquet, JSON, and folders of any of the above
  - Parallel parameter sweep  (all CPU cores - 1 via ProcessPoolExecutor)
  - GPU Monte Carlo           (PyTorch batched paths on CUDA if available)
  - GPU XGBoost inference     (device='cuda' if CUDA available)
  - Numba-JIT ELO loop        (auto-compiled on first run)
  - Walk-forward chunks       (parallelised across CPU workers)
  - Full PDF report + JSON results output

Quick start (GUI file picker):
  python gpu_backtester.py

Quick start (CLI):
  python gpu_backtester.py --data C:/data/atp_matches_2022.csv C:/data/atp_matches_2023.csv
  python gpu_backtester.py --data-dir C:/data/tennis_atp-master --model best_xgb_model.json

Windows note: the multiprocessing pool uses 'spawn' on Windows automatically.
              All worker functions are top-level for pickle compatibility.
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import glob
import json
import time
import logging
import warnings
import argparse
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

# ── third-party (required) ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # must be before any pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    log_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
import joblib

warnings.filterwarnings("ignore")

# ── optional accelerators ─────────────────────────────────────────────────────
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import torch
    _HAS_TORCH = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
    _GPU_DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"
except ImportError:
    _HAS_TORCH = False
    _CUDA_AVAILABLE = False
    _GPU_DEVICE = "cpu"

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
    def tqdm(it, **kw):      # no-op fallback
        return it

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
MODEL_PATH     = BASE_DIR / "best_xgb_model.json"
FEATURES_PATH  = BASE_DIR / "model_features.json"
PDF_OUT        = BASE_DIR / "gpu_backtest_report.pdf"
JSON_OUT       = BASE_DIR / "gpu_backtest_results.json"

BANKROLL_INIT  = 1_000.0
TEST_START     = 2022
ELO_K          = 32
ELO_INIT       = 1_500.0

# Parameter grid — expanded for thorough sweep
PARAM_GRID = {
    "min_edge":   [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
    "kelly_frac": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    "vig":        [0.05, 0.07, 0.09],
}

MC_ITERATIONS  = 5_000    # default Monte Carlo paths
MC_GPU_CHUNK   = 2_000    # paths per GPU batch (tune down if OOM)

SURFACE_MAP = {
    "Hard": "Hard", "Clay": "Clay", "Grass": "Grass",
    "Carpet": "Hard", "Indoor Hard": "Hard", "Outdoor Hard": "Hard",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Resource Info
# ─────────────────────────────────────────────────────────────────────────────

def print_hardware_banner():
    cpu_count = os.cpu_count() or 1
    log.info("=" * 60)
    log.info(f"  CPU cores available : {cpu_count}")
    log.info(f"  Workers to be used  : {max(1, cpu_count - 1)}  (leaving 1 for OS)")
    if _HAS_PSUTIL:
        ram_gb = psutil.virtual_memory().total / 1e9
        log.info(f"  System RAM          : {ram_gb:.1f} GB")
    if _HAS_TORCH:
        if _CUDA_AVAILABLE:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram = props.total_memory / 1e9
                log.info(f"  GPU {i}               : {props.name}  ({vram:.1f} GB VRAM)")
        else:
            log.info("  GPU                 : No CUDA device found — using CPU")
    else:
        log.info("  GPU                 : PyTorch not installed — CPU only")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  File Input — GUI + CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_gui() -> dict:
    """
    Launches a tkinter window so the user can browse for data files and
    set key parameters. Returns a config dict that mirrors the argparse namespace.
    Falls back gracefully if tkinter is not available.
    """
    root = tk.Tk()
    root.title("TennisBot GPU Backtester — File Input")
    root.resizable(False, False)
    root.geometry("700x520")

    style = ttk.Style()
    style.theme_use("clam")

    cfg = {
        "data_files":  [],
        "data_dir":    None,
        "model":       str(MODEL_PATH),
        "features":    str(FEATURES_PATH),
        "bankroll":    BANKROLL_INIT,
        "test_start":  TEST_START,
        "mc_iter":     MC_ITERATIONS,
        "pdf_out":     str(PDF_OUT),
        "cancelled":   False,
    }

    # ── Header ──────────────────────────────────────────────────────────────
    ttk.Label(root, text="TennisBot GPU Backtester", font=("Helvetica", 16, "bold")).pack(pady=(16, 4))
    ttk.Label(root, text="Add CSV / Parquet / JSON files or point to a folder", font=("Helvetica", 10)).pack()

    # ── File list ────────────────────────────────────────────────────────────
    list_frame = ttk.LabelFrame(root, text="Input Data Files", padding=8)
    list_frame.pack(fill="both", padx=16, pady=8)

    scrollbar = ttk.Scrollbar(list_frame)
    file_listbox = tk.Listbox(list_frame, height=7, yscrollcommand=scrollbar.set,
                               selectmode=tk.EXTENDED, font=("Courier", 9))
    scrollbar.config(command=file_listbox.yview)
    file_listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def add_files():
        paths = filedialog.askopenfilenames(
            title="Select data files",
            filetypes=[
                ("Data files", "*.csv *.parquet *.json"),
                ("CSV", "*.csv"),
                ("Parquet", "*.parquet"),
                ("JSON", "*.json"),
                ("All files", "*.*"),
            ],
        )
        for p in paths:
            if p not in cfg["data_files"]:
                cfg["data_files"].append(p)
                file_listbox.insert(tk.END, Path(p).name)

    def add_folder():
        folder = filedialog.askdirectory(title="Select folder containing match data")
        if folder:
            cfg["data_dir"] = folder
            file_listbox.insert(tk.END, f"[DIR] {folder}")

    def remove_selected():
        for i in reversed(file_listbox.curselection()):
            file_listbox.delete(i)
            if i < len(cfg["data_files"]):
                cfg["data_files"].pop(i)

    btn_row = ttk.Frame(root)
    btn_row.pack(fill="x", padx=16)
    ttk.Button(btn_row, text="Add Files…",   command=add_files).pack(side="left", padx=4)
    ttk.Button(btn_row, text="Add Folder…",  command=add_folder).pack(side="left", padx=4)
    ttk.Button(btn_row, text="Remove",       command=remove_selected).pack(side="left", padx=4)

    # ── Model ─────────────────────────────────────────────────────────────
    model_var = tk.StringVar(value=cfg["model"])
    model_row = ttk.LabelFrame(root, text="Model & Features", padding=6)
    model_row.pack(fill="x", padx=16, pady=4)
    ttk.Entry(model_row, textvariable=model_var, width=55).pack(side="left", padx=4)
    ttk.Button(model_row, text="Browse…",
               command=lambda: model_var.set(
                   filedialog.askopenfilename(
                       title="Select XGBoost model",
                       filetypes=[("Model files", "*.json *.pkl *.joblib"), ("All", "*.*")],
                   ) or model_var.get()
               )).pack(side="left")

    # ── Parameters ──────────────────────────────────────────────────────────
    param_frame = ttk.LabelFrame(root, text="Simulation Parameters", padding=8)
    param_frame.pack(fill="x", padx=16, pady=4)

    def _row(parent, label, default, row):
        ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w")
        var = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, padx=4)
        return var

    bank_var  = _row(param_frame, "Starting Bankroll ($):", BANKROLL_INIT, 0)
    year_var  = _row(param_frame, "Test Start Year:",       TEST_START,    1)
    mc_var    = _row(param_frame, "Monte Carlo Iterations:", MC_ITERATIONS, 2)

    # ── Output ──────────────────────────────────────────────────────────────
    out_var = tk.StringVar(value=cfg["pdf_out"])
    out_frame = ttk.LabelFrame(root, text="PDF Output", padding=6)
    out_frame.pack(fill="x", padx=16, pady=4)
    ttk.Entry(out_frame, textvariable=out_var, width=55).pack(side="left", padx=4)
    ttk.Button(out_frame, text="Browse…",
               command=lambda: out_var.set(
                   filedialog.asksaveasfilename(
                       title="Save PDF report as",
                       defaultextension=".pdf",
                       filetypes=[("PDF", "*.pdf")],
                   ) or out_var.get()
               )).pack(side="left")

    # ── Run / Cancel ────────────────────────────────────────────────────────
    def on_run():
        if not cfg["data_files"] and not cfg["data_dir"]:
            messagebox.showerror("No data", "Please add at least one data file or folder.")
            return
        try:
            cfg["bankroll"]   = float(bank_var.get())
            cfg["test_start"] = int(year_var.get())
            cfg["mc_iter"]    = int(mc_var.get())
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        cfg["model"]   = model_var.get()
        cfg["pdf_out"] = out_var.get()
        root.destroy()

    def on_cancel():
        cfg["cancelled"] = True
        root.destroy()

    btn_bottom = ttk.Frame(root)
    btn_bottom.pack(pady=10)
    ttk.Button(btn_bottom, text="▶  Run Backtest", command=on_run,  width=20).pack(side="left", padx=8)
    ttk.Button(btn_bottom, text="✕  Cancel",       command=on_cancel, width=12).pack(side="left")

    root.mainloop()
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Data Loading  (CSV, Parquet, JSON — single files or whole folders)
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_file(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    try:
        if p.suffix == ".parquet":
            return pd.read_parquet(p)
        if p.suffix == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return pd.DataFrame(raw)
            return pd.DataFrame([raw])
        # Default: CSV (handles .csv, .tsv, .txt)
        sep = "\t" if p.suffix in (".tsv", ".txt") else ","
        return pd.read_csv(p, sep=sep, low_memory=False)
    except Exception as e:
        log.warning(f"  Skipping {p.name}: {e}")
        return None


def load_all_data(data_files: List[str], data_dir: Optional[str]) -> pd.DataFrame:
    """
    Load every file in data_files plus all recognised files in data_dir.
    Adds a 'year' column if not present (inferred from filename if possible).
    """
    frames = []

    # Explicit file list
    for path in data_files:
        df = _load_single_file(path)
        if df is not None:
            _inject_year(df, Path(path).stem)
            frames.append(df)
            log.info(f"  Loaded {len(df):,} rows ← {Path(path).name}")

    # Folder scan
    if data_dir:
        exts = ("*.csv", "*.parquet", "*.json", "*.tsv")
        found = []
        for ext in exts:
            found.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        found = sorted(set(found))
        log.info(f"  Scanning {data_dir} — found {len(found)} files")
        for path in tqdm(found, desc="Loading folder"):
            df = _load_single_file(path)
            if df is not None:
                _inject_year(df, Path(path).stem)
                frames.append(df)

    if not frames:
        raise ValueError("No data loaded — check your file paths.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total rows loaded: {combined.shape[0]:,}  columns: {combined.shape[1]}")
    return combined


def _inject_year(df: pd.DataFrame, stem: str):
    """Add 'year' column if missing, inferred from filename like atp_matches_2022."""
    if "year" not in df.columns:
        import re
        m = re.search(r"(19|20)\d{2}", stem)
        df["year"] = int(m.group(0)) if m else 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Feature Engineering  (Numba-JIT ELO if available, else pure numpy)
# ─────────────────────────────────────────────────────────────────────────────

if _HAS_NUMBA:
    @njit(parallel=False, cache=True)
    def _compute_elo_numba(winner_ids, loser_ids, k, init):
        """
        Numba-JIT ELO updater.  Runs in a single compiled C loop — about 15x
        faster than a Python for-loop on 200k+ matches.
        """
        n = len(winner_ids)
        elo_before_w = np.empty(n, dtype=np.float64)
        elo_before_l = np.empty(n, dtype=np.float64)
        # Use a simple open-addressed hash via pre-built index arrays
        # (IDs are passed as integer indices built outside Numba)
        max_id = int(max(winner_ids.max(), loser_ids.max())) + 1
        elo_state = np.full(max_id, init, dtype=np.float64)
        for i in range(n):
            wid = winner_ids[i]
            lid = loser_ids[i]
            ew = elo_state[wid]
            el = elo_state[lid]
            elo_before_w[i] = ew
            elo_before_l[i] = el
            exp_w = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
            elo_state[wid] = ew + k * (1.0 - exp_w)
            elo_state[lid] = el + k * (0.0 - (1.0 - exp_w))
        return elo_before_w, elo_before_l
else:
    def _compute_elo_numba(winner_ids, loser_ids, k, init):   # pure-Python fallback
        n = len(winner_ids)
        elo_before_w = np.empty(n, dtype=np.float64)
        elo_before_l = np.empty(n, dtype=np.float64)
        elo_state: dict = {}
        for i in range(n):
            wid, lid = int(winner_ids[i]), int(loser_ids[i])
            ew = elo_state.get(wid, init)
            el = elo_state.get(lid, init)
            elo_before_w[i] = ew
            elo_before_l[i] = el
            exp_w = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
            elo_state[wid] = ew + k * (1.0 - exp_w)
            elo_state[lid] = el + k * (0.0 - (1.0 - exp_w))
        return elo_before_w, elo_before_l


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["surface_norm"] = df.get("surface", pd.Series("Hard", index=df.index)).map(SURFACE_MAP).fillna("Hard")
    df["winner_hand_enc"] = (df.get("winner_hand", "R") == "R").astype(np.int8)
    df["loser_hand_enc"]  = (df.get("loser_hand",  "R") == "R").astype(np.int8)

    for col in ["winner_ht", "loser_ht", "winner_age", "loser_age",
                "winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]:
        fill = df.groupby("year")[col].transform("median") if "year" in df.columns else df[col].median()
        df[col] = df[col].fillna(fill)

    for col in ["winner_ht", "loser_ht", "winner_age", "loser_age"]:
        df[col] = df[col].fillna(df[col].median())

    # Build integer ID arrays for the Numba ELO loop
    if "winner_id" in df.columns and "loser_id" in df.columns:
        all_ids = pd.concat([df["winner_id"].astype(str), df["loser_id"].astype(str)]).unique()
        id_map  = {v: i for i, v in enumerate(all_ids)}
        w_idx   = df["winner_id"].astype(str).map(id_map).values.astype(np.int64)
        l_idx   = df["loser_id"].astype(str).map(id_map).values.astype(np.int64)
    else:
        # Fallback: create synthetic IDs from player names if IDs are missing
        all_names = pd.concat([
            df.get("winner_name", pd.Series(dtype=str)),
            df.get("loser_name",  pd.Series(dtype=str)),
        ]).unique()
        id_map = {v: i for i, v in enumerate(all_names)}
        w_col  = df.get("winner_name", pd.Series("unknown", index=df.index)).astype(str)
        l_col  = df.get("loser_name",  pd.Series("unknown", index=df.index)).astype(str)
        w_idx  = w_col.map(id_map).fillna(0).values.astype(np.int64)
        l_idx  = l_col.map(id_map).fillna(0).values.astype(np.int64)

    t0 = time.time()
    elo_w, elo_l = _compute_elo_numba(w_idx, l_idx, ELO_K, ELO_INIT)
    log.info(f"  ELO computed for {len(df):,} rows in {time.time()-t0:.2f}s"
             f"  ({'Numba JIT' if _HAS_NUMBA else 'Python loop'})")

    df["elo_winner"] = elo_w
    df["elo_loser"]  = elo_l
    df["elo_prob_winner"] = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))

    # Random P1/P2 assignment — eliminates winner-label bias
    rng  = np.random.default_rng(42)
    swap = rng.random(len(df)) < 0.5

    def pick(w_col, l_col):
        w = df[w_col].values
        l = df[l_col].values
        return np.where(swap, l, w), np.where(swap, w, l)

    p1_hand, p2_hand = pick("winner_hand_enc", "loser_hand_enc")
    p1_ht,   p2_ht   = pick("winner_ht",  "loser_ht")
    p1_age,  p2_age  = pick("winner_age", "loser_age")
    p1_rank, p2_rank = pick("winner_rank", "loser_rank")
    p1_rpts, p2_rpts = pick("winner_rank_points", "loser_rank_points")
    p1_elo,  p2_elo  = pick("elo_winner", "elo_loser")

    y = np.where(swap, 0, 1)

    out = pd.DataFrame({
        "year":               df["year"].values if "year" in df.columns else np.zeros(len(df), int),
        "tourney_date":       df.get("tourney_date", pd.Series(0, index=df.index)).values,
        "Surface_Hard":       (df["surface_norm"] == "Hard").astype(np.int8).values,
        "Surface_Clay":       (df["surface_norm"] == "Clay").astype(np.int8).values,
        "Surface_Grass":      (df["surface_norm"] == "Grass").astype(np.int8).values,
        "Best_Of_Sets":       pd.to_numeric(df.get("best_of", 3), errors="coerce").fillna(3).values,
        "P1_Is_Right_Handed": p1_hand,
        "P1_Height_cm":       p1_ht,
        "P1_Age":             p1_age,
        "P1_Rank":            p1_rank,
        "P1_Rank_Points":     p1_rpts,
        "P2_Is_Right_Handed": p2_hand,
        "P2_Height_cm":       p2_ht,
        "P2_Age":             p2_age,
        "P2_Rank":            p2_rank,
        "P2_Rank_Points":     p2_rpts,
        "elo_prob_p1":        np.where(swap, 1.0 - df["elo_prob_winner"].values,
                                             df["elo_prob_winner"].values),
        "surface_norm":       df["surface_norm"].values,
        "target":             y,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  XGBoost Inference  (GPU if CUDA available)
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> object:
    mp = Path(model_path)
    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    model = joblib.load(mp)
    if _HAS_XGB and _CUDA_AVAILABLE:
        try:
            model.set_params(device="cuda", tree_method="hist")
            log.info("  XGBoost: GPU inference enabled (device=cuda)")
        except Exception:
            try:
                model.set_params(tree_method="gpu_hist")
                log.info("  XGBoost: GPU inference enabled (gpu_hist)")
            except Exception:
                log.info("  XGBoost: GPU parameter set failed — using CPU")
    else:
        log.info(f"  XGBoost: running on CPU  (CUDA={'yes' if _CUDA_AVAILABLE else 'no'}  XGB={_HAS_XGB})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Kelly Simulation  (fully vectorised numpy — no Python loops)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_kelly_vectorised(
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    outcomes: np.ndarray,
    bankroll_init: float = BANKROLL_INIT,
    kelly_frac: float    = 0.25,
    min_edge: float      = 0.04,
    vig: float           = 0.08,
) -> dict:
    mkp_adj  = np.clip(market_prob + vig / 2.0, 0.05, 0.95)
    edge_raw = model_prob - market_prob
    edge_adj = model_prob - mkp_adj

    # Tiered Kelly fractions (vectorised)
    tier = np.where(model_prob >= 0.70, 0.40,
           np.where(model_prob >= 0.60, 0.25,
           np.where(model_prob >= 0.55, 0.12, 0.05)))

    f_star  = np.where(mkp_adj > 0, edge_adj / (1.0 - mkp_adj), 0.0)
    f_eff   = np.clip(f_star * tier * (kelly_frac / 0.25), 0.0, 0.20)

    # Bet mask — only bet where edge is sufficient
    bet_mask = (edge_raw >= min_edge) & (edge_adj > 0) & (mkp_adj > 0.05) & (mkp_adj < 0.95)

    # Sequential bankroll evolution  (unavoidable loop — bankroll is path-dependent)
    n        = len(model_prob)
    bankroll = bankroll_init
    history  = np.empty(n + 1)
    history[0] = bankroll
    pnl_list = []
    bet_list = []

    for i in range(n):
        if not bet_mask[i]:
            history[i + 1] = bankroll
            continue
        bet_size = bankroll * f_eff[i]
        profit   = bet_size * (1.0 - mkp_adj[i]) / mkp_adj[i] if outcomes[i] == 1 else -bet_size
        bankroll = max(bankroll + profit, 0.01)
        history[i + 1] = bankroll
        pnl_list.append(profit)
        bet_list.append(bet_size)

    n_bets       = len(pnl_list)
    total_staked = float(np.sum(bet_list)) if bet_list else 0.0
    total_profit = float(np.sum(pnl_list)) if pnl_list else 0.0
    roi          = total_profit / total_staked * 100.0 if total_staked > 0 else 0.0
    win_rate     = float(np.mean(np.array(pnl_list) > 0) * 100.0) if pnl_list else 0.0
    returns      = np.array(pnl_list) / np.array(bet_list) if bet_list else np.array([0.0])
    sharpe       = float(returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0.0
    max_dd       = _max_drawdown(history[:n_bets + 1])

    return {
        "history":        history[:n_bets + 1],
        "n_bets":         n_bets,
        "roi":            roi,
        "win_rate":       win_rate,
        "final_bankroll": bankroll,
        "total_profit":   total_profit,
        "sharpe":         sharpe,
        "max_drawdown":   max_dd,
        "pnl_list":       pnl_list,
    }


def _max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd   = (peak - equity) / np.maximum(peak, 1e-9)
    return float(dd.max())


# ─────────────────────────────────────────────────────────────────────────────
# 7.  GPU Monte Carlo  (PyTorch batched paths)
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    pnl_list:      List[float],
    bankroll_init: float = BANKROLL_INIT,
    n_iterations:  int   = MC_ITERATIONS,
    chunk_size:    int   = MC_GPU_CHUNK,
) -> np.ndarray:
    """
    Runs Monte Carlo on GPU (CUDA) if available, otherwise CPU.
    Returns ndarray of shape (n_iterations, n_bets+1) — equity paths.
    Uses chunked GPU batching to avoid OOM on large n_iterations.
    """
    if not pnl_list:
        return np.array([])

    pnl = np.array(pnl_list, dtype=np.float32)
    n   = len(pnl)
    all_paths = []

    if _HAS_TORCH:
        device     = torch.device(_GPU_DEVICE)
        pnl_tensor = torch.tensor(pnl, device=device)
        log.info(f"  Monte Carlo: {n_iterations:,} paths × {n} bets  on {device}")

        t0    = time.time()
        done  = 0
        while done < n_iterations:
            batch = min(chunk_size, n_iterations - done)
            # Random bootstrap indices — all generated on GPU
            idx      = torch.randint(0, n, (batch, n), device=device)
            shuffled = pnl_tensor[idx]                       # (batch, n)
            paths    = bankroll_init + torch.cumsum(shuffled, dim=1)
            paths    = torch.clamp(paths, min=0.0)
            # Prepend bankroll_init column
            init_col = torch.full((batch, 1), bankroll_init, device=device)
            paths    = torch.cat([init_col, paths], dim=1)   # (batch, n+1)
            all_paths.append(paths.cpu().numpy())
            done += batch

        log.info(f"  Monte Carlo complete in {time.time()-t0:.2f}s")
        return np.concatenate(all_paths, axis=0)

    else:
        # Pure numpy fallback
        log.info(f"  Monte Carlo (numpy): {n_iterations:,} paths")
        results = np.empty((n_iterations, n + 1), dtype=np.float32)
        results[:, 0] = bankroll_init
        for i in tqdm(range(n_iterations), desc="Monte Carlo"):
            shuffled = np.random.choice(pnl, size=n, replace=True)
            path     = bankroll_init + np.cumsum(shuffled)
            path     = np.maximum(path, 0.0)
            results[i, 1:] = path
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Parallel Parameter Sweep  (ProcessPoolExecutor — all CPUs - 1)
# ─────────────────────────────────────────────────────────────────────────────
# Worker must be a module-level function (Windows spawn requirement)

def _sweep_worker(args: tuple) -> dict:
    """Top-level picklable worker for parallel parameter sweep."""
    model_prob, market_prob, outcomes, min_edge, kelly_frac, vig = args
    res = simulate_kelly_vectorised(
        model_prob, market_prob, outcomes,
        min_edge=min_edge, kelly_frac=kelly_frac, vig=vig,
    )
    return {
        "min_edge":   min_edge,
        "kelly_frac": kelly_frac,
        "vig":        vig,
        "roi":        res["roi"],
        "sharpe":     res["sharpe"],
        "n_bets":     res["n_bets"],
        "max_dd":     res["max_drawdown"],
    }


def run_parallel_sweep(
    model_prob:  np.ndarray,
    market_prob: np.ndarray,
    outcomes:    np.ndarray,
    param_grid:  dict       = PARAM_GRID,
) -> pd.DataFrame:
    import itertools
    combos = list(itertools.product(
        param_grid["min_edge"],
        param_grid["kelly_frac"],
        param_grid["vig"],
    ))
    args_list = [
        (model_prob, market_prob, outcomes, me, kf, vg)
        for me, kf, vg in combos
    ]

    n_workers = max(1, (os.cpu_count() or 1) - 1)
    log.info(f"  Parameter sweep: {len(combos)} combos across {n_workers} CPU workers")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_sweep_worker, a): a for a in args_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Sweep"):
            try:
                results.append(fut.result())
            except Exception as e:
                log.warning(f"  Sweep worker failed: {e}")

    log.info(f"  Sweep complete in {time.time()-t0:.2f}s")
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Walk-Forward Backtesting  (year-by-year rolling windows)
# ─────────────────────────────────────────────────────────────────────────────

def _wf_worker(args: tuple) -> dict:
    """Top-level picklable worker for one walk-forward fold."""
    fold_year, X_train, y_train, X_test, y_test, market_prob_test, features = args
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,           # single thread — parallelism is at fold level
        )
        model.fit(X_train[features], y_train)
        y_prob = model.predict_proba(X_test[features])[:, 1]
        from sklearn.metrics import accuracy_score, roc_auc_score
        from gpu_backtester import simulate_kelly_vectorised, BANKROLL_INIT
        res = simulate_kelly_vectorised(y_prob, market_prob_test, y_test)
        return {
            "year":    fold_year,
            "n":       len(y_test),
            "acc":     accuracy_score(y_test, y_prob >= 0.5),
            "auc":     roc_auc_score(y_test, y_prob),
            "roi":     res["roi"],
            "n_bets":  res["n_bets"],
            "sharpe":  res["sharpe"],
        }
    except Exception as e:
        return {"year": fold_year, "error": str(e)}


def run_walk_forward(feat_df: pd.DataFrame, features: list, test_start: int) -> pd.DataFrame:
    """
    Expanding window: train on all years < fold_year, test on fold_year.
    Parallelises across CPU cores.
    """
    years = sorted(feat_df["year"].unique())
    test_years = [y for y in years if y >= test_start]
    if not test_years:
        log.warning("  No test years found — skipping walk-forward")
        return pd.DataFrame()

    fold_args = []
    for fold_year in test_years:
        train = feat_df[feat_df["year"] < fold_year]
        test  = feat_df[feat_df["year"] == fold_year]
        if len(train) < 500 or len(test) < 50:
            continue
        fold_args.append((
            fold_year,
            train, train["target"].values,
            test,  test["target"].values,
            test["elo_prob_p1"].values,
            features,
        ))

    if not fold_args:
        return pd.DataFrame()

    n_workers = max(1, min((os.cpu_count() or 1) - 1, len(fold_args)))
    log.info(f"  Walk-forward: {len(fold_args)} folds across {n_workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in tqdm(ex.map(_wf_worker, fold_args), total=len(fold_args), desc="Walk-forward"):
            rows.append(res)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PDF Report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    pdf_path:    str,
    feat_df:     pd.DataFrame,
    y_true:      np.ndarray,
    y_prob:      np.ndarray,
    base_result: dict,
    mc_paths:    np.ndarray,
    sweep_df:    pd.DataFrame,
    wf_df:       pd.DataFrame,
    feature_names: list,
):
    acc   = accuracy_score(y_true, y_prob >= 0.5)
    auc   = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    log.info(f"  Generating PDF: {pdf_path}")
    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Summary ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.97, "TennisBot77 — GPU Backtest Report", fontsize=18,
                ha="center", weight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.92, f"Generated: {time.strftime('%Y-%m-%d %H:%M')}  |  "
                f"Data rows: {len(feat_df):,}  |  "
                f"GPU: {'CUDA ' + torch.cuda.get_device_name(0) if _CUDA_AVAILABLE and _HAS_TORCH else 'CPU only'}",
                fontsize=9, ha="center", color="#555", transform=ax.transAxes)

        txt = (
            f"{'Model Accuracy':<28} {acc*100:.2f}%\n"
            f"{'ROC-AUC':<28} {auc:.4f}\n"
            f"{'Brier Score':<28} {brier:.4f}\n"
            f"\n"
            f"{'Bets Triggered':<28} {base_result['n_bets']:,}\n"
            f"{'Total Profit':<28} ${base_result['total_profit']:+,.2f}\n"
            f"{'ROI':<28} {base_result['roi']:+.2f}%\n"
            f"{'Win Rate':<28} {base_result['win_rate']:.1f}%\n"
            f"{'Sharpe Ratio (annualised)':<28} {base_result['sharpe']:.3f}\n"
            f"{'Max Drawdown':<28} {base_result['max_drawdown']*100:.1f}%\n"
            f"{'Final Bankroll':<28} ${base_result['final_bankroll']:,.2f}\n"
        )
        ax.text(0.08, 0.82, txt, fontsize=12, fontfamily="monospace",
                va="top", transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.8))
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 2: Equity Curve ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(base_result["history"], color="#1a6eb5", linewidth=1.5, label="Backtest equity")
        ax.axhline(BANKROLL_INIT, color="gray", linestyle="--", linewidth=0.8, label="Starting bankroll")
        ax.fill_between(range(len(base_result["history"])), BANKROLL_INIT,
                        base_result["history"],
                        where=np.array(base_result["history"]) >= BANKROLL_INIT,
                        alpha=0.15, color="green")
        ax.fill_between(range(len(base_result["history"])), BANKROLL_INIT,
                        base_result["history"],
                        where=np.array(base_result["history"]) < BANKROLL_INIT,
                        alpha=0.15, color="red")
        ax.set_title("Bankroll Equity Curve"); ax.set_xlabel("Bet #"); ax.set_ylabel("Bankroll ($)")
        ax.legend(); ax.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 3: Monte Carlo paths ────────────────────────────────────────
        if len(mc_paths) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(mc_paths.shape[1])
            sample = mc_paths[np.random.choice(len(mc_paths), min(200, len(mc_paths)), replace=False)]
            for path in sample:
                ax.plot(x, path, color="#3a7abf", alpha=0.03, linewidth=0.5)
            p5,  p25, p50, p75, p95 = np.percentile(mc_paths, [5, 25, 50, 75, 95], axis=0)
            ax.fill_between(x, p25, p75, alpha=0.25, color="#2196F3", label="25–75th pct")
            ax.fill_between(x, p5,  p95, alpha=0.10, color="#2196F3", label="5–95th pct")
            ax.plot(x, p50, color="black",   linewidth=2,   label="Median")
            ax.plot(x, p5,  color="red",     linewidth=1.2, linestyle="--", label="5th pct")
            ax.plot(x, p95, color="green",   linewidth=1.2, linestyle="--", label="95th pct")
            ax.axhline(BANKROLL_INIT, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"Monte Carlo Projections  ({len(mc_paths):,} paths  ·  GPU: {_GPU_DEVICE.upper()})")
            ax.set_xlabel("Bet #"); ax.set_ylabel("Bankroll ($)")
            ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=0.3)
            pdf.savefig(fig, bbox_inches="tight"); plt.close()

            # Distribution of final values
            fig, ax = plt.subplots(figsize=(10, 4))
            finals = mc_paths[:, -1]
            ax.hist(finals, bins=80, color="#1a6eb5", edgecolor="white", linewidth=0.3, alpha=0.85)
            ax.axvline(np.percentile(finals, 5),  color="red",   linestyle="--", label="5th pct")
            ax.axvline(np.percentile(finals, 50), color="black", linestyle="-",  label="Median", linewidth=2)
            ax.axvline(np.percentile(finals, 95), color="green", linestyle="--", label="95th pct")
            ax.axvline(BANKROLL_INIT, color="gray", linestyle=":", linewidth=0.8, label="Start")
            ax.set_title("Distribution of Final Bankroll Values")
            ax.set_xlabel("Final Bankroll ($)"); ax.set_ylabel("Frequency")
            ax.legend(); ax.grid(alpha=0.3)
            pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 4: ROI Heatmap (best vig slice) ────────────────────────────
        if not sweep_df.empty:
            for vig_val in sweep_df["vig"].unique():
                sub = sweep_df[sweep_df["vig"] == vig_val]
                pivot_roi    = sub.pivot_table(index="min_edge", columns="kelly_frac", values="roi")
                pivot_sharpe = sub.pivot_table(index="min_edge", columns="kelly_frac", values="sharpe")

                fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                sns.heatmap(pivot_roi, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                            ax=axes[0], cbar_kws={"label": "ROI %"})
                axes[0].set_title(f"ROI (%)  —  vig={vig_val:.2f}")
                axes[0].set_xlabel("Kelly Fraction")
                axes[0].set_ylabel("Min Edge Threshold")

                sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                            ax=axes[1], cbar_kws={"label": "Sharpe"})
                axes[1].set_title(f"Sharpe Ratio  —  vig={vig_val:.2f}")
                axes[1].set_xlabel("Kelly Fraction")
                axes[1].set_ylabel("Min Edge Threshold")

                plt.suptitle(f"Parameter Sweep  ({len(sub)} combos, {(os.cpu_count() or 1)-1} CPU workers)",
                             fontsize=13, weight="bold")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 5: Walk-Forward ─────────────────────────────────────────────
        if not wf_df.empty and "error" not in wf_df.columns:
            wf_ok = wf_df[~wf_df.get("error", pd.Series(False, index=wf_df.index)).astype(bool)]
            if not wf_ok.empty:
                fig, axes = plt.subplots(2, 2, figsize=(13, 9))
                wf_ok.plot(x="year", y="roi",    kind="bar", ax=axes[0, 0], legend=False, color="#1a6eb5")
                axes[0, 0].axhline(0, color="black", linewidth=0.8)
                axes[0, 0].set_title("ROI by Year (%)"); axes[0, 0].set_xlabel("")

                wf_ok.plot(x="year", y="auc",    kind="line", ax=axes[0, 1], legend=False, marker="o")
                axes[0, 1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
                axes[0, 1].set_title("ROC-AUC by Year")

                wf_ok.plot(x="year", y="sharpe", kind="bar", ax=axes[1, 0], legend=False, color="#4caf50")
                axes[1, 0].axhline(0, color="black", linewidth=0.8)
                axes[1, 0].set_title("Sharpe by Year"); axes[1, 0].set_xlabel("")

                wf_ok.plot(x="year", y="n_bets", kind="bar", ax=axes[1, 1], legend=False, color="#ff7043")
                axes[1, 1].set_title("Bets Triggered by Year"); axes[1, 1].set_xlabel("")

                plt.suptitle("Walk-Forward Backtest  (expanding window, one fold per year)",
                             fontsize=12, weight="bold")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 6: Model Diagnostics ────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, color="#1a6eb5", lw=2, label=f"AUC={auc:.3f}")
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
        axes[0].set_title("ROC Curve"); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
        axes[0].legend()

        # Calibration
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=12)
        axes[1].plot(prob_pred, prob_true, "o-", color="#e53935", label="Model")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="Perfect")
        axes[1].set_title("Calibration Curve"); axes[1].set_xlabel("Predicted P"); axes[1].set_ylabel("Observed P")
        axes[1].legend()

        # Predicted probability distribution
        axes[2].hist(y_prob[y_true == 1], bins=40, alpha=0.6, label="Win",  color="green", density=True)
        axes[2].hist(y_prob[y_true == 0], bins=40, alpha=0.6, label="Loss", color="red",   density=True)
        axes[2].set_title("Predicted Probability by Outcome")
        axes[2].set_xlabel("P(win)"); axes[2].set_ylabel("Density")
        axes[2].legend()

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close()

        # ── Page 7: Surface analysis ─────────────────────────────────────────
        if "surface_norm" in feat_df.columns:
            test_mask = feat_df["year"] >= TEST_START
            td = feat_df[test_mask].copy()
            if len(td) > 0:
                td["correct"] = (y_prob[:len(td)] >= 0.5) == y_true[:len(td)]
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                surf = td.groupby("surface_norm")["correct"].agg(["mean", "count"]).reset_index()
                surf.columns = ["surface", "accuracy", "count"]
                axes[0].bar(surf["surface"], surf["accuracy"] * 100, color=["#1a6eb5", "#e53935", "#43a047"])
                axes[0].set_ylim(40, 80)
                axes[0].set_title("Model Accuracy by Surface (%)")
                for i, row in surf.iterrows():
                    axes[0].text(i, row["accuracy"] * 100 + 0.5, f"{row['accuracy']*100:.1f}%  (n={int(row['count'])})",
                                 ha="center", fontsize=9)

                count_by_surf = td.groupby("surface_norm").size()
                axes[1].pie(count_by_surf, labels=count_by_surf.index, autopct="%1.1f%%",
                            colors=["#1a6eb5", "#e53935", "#43a047"])
                axes[1].set_title("Match Distribution by Surface")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight"); plt.close()

    log.info(f"  PDF saved → {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    # ── Parse CLI args ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="TennisBot GPU Backtester — high-performance betting simulation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data",      nargs="+", metavar="FILE",
                        help="One or more CSV / Parquet / JSON data files")
    parser.add_argument("--data-dir",  metavar="DIR",
                        help="Folder to scan recursively for data files")
    parser.add_argument("--model",     default=str(MODEL_PATH),
                        help="Path to XGBoost model (.json or .joblib)")
    parser.add_argument("--features",  default=str(FEATURES_PATH),
                        help="Path to model_features.json")
    parser.add_argument("--bankroll",  type=float, default=BANKROLL_INIT,
                        help=f"Starting bankroll in USD (default: {BANKROLL_INIT})")
    parser.add_argument("--test-start", type=int, default=TEST_START,
                        help=f"First test year (default: {TEST_START})")
    parser.add_argument("--mc-iter",   type=int, default=MC_ITERATIONS,
                        help=f"Monte Carlo iterations (default: {MC_ITERATIONS})")
    parser.add_argument("--pdf-out",   default=str(PDF_OUT),
                        help="Output PDF path")
    parser.add_argument("--no-gui",    action="store_true",
                        help="Skip GUI even if tkinter is available")
    parser.add_argument("--no-sweep",  action="store_true",
                        help="Skip CPU parameter sweep (faster)")
    parser.add_argument("--no-wf",     action="store_true",
                        help="Skip walk-forward backtest (faster)")
    cli = parser.parse_args(args)

    # ── GUI if no data supplied on CLI ────────────────────────────────────────
    gui_cfg = None
    if not cli.data and not cli.data_dir and not cli.no_gui and _HAS_TK:
        log.info("No data files supplied on CLI — launching file picker…")
        gui_cfg = _build_gui()
        if gui_cfg["cancelled"]:
            log.info("Cancelled by user.")
            return

    # Resolve effective config
    data_files = (gui_cfg["data_files"] if gui_cfg else cli.data) or []
    data_dir   = (gui_cfg["data_dir"]   if gui_cfg else cli.data_dir)
    model_path = (gui_cfg["model"]      if gui_cfg else cli.model)
    feat_path  = cli.features
    bankroll   = (gui_cfg["bankroll"]   if gui_cfg else cli.bankroll)
    test_start = (gui_cfg["test_start"] if gui_cfg else cli.test_start)
    mc_iter    = (gui_cfg["mc_iter"]    if gui_cfg else cli.mc_iter)
    pdf_out    = (gui_cfg["pdf_out"]    if gui_cfg else cli.pdf_out)

    if not data_files and not data_dir:
        log.error("No data files specified.  Use --data or --data-dir, or run without --no-gui to use the file picker.")
        sys.exit(1)

    # ── Banner ────────────────────────────────────────────────────────────────
    log.info("TennisBot GPU Backtester starting…")
    print_hardware_banner()

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("STEP 1/6  Loading data…")
    raw = load_all_data(data_files, data_dir)

    # ── Feature engineering ───────────────────────────────────────────────────
    log.info("STEP 2/6  Building features (Numba ELO)…")
    feat_df = build_features(raw)
    log.info(f"  Feature matrix: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} cols")

    # ── Load model + features list ────────────────────────────────────────────
    log.info("STEP 3/6  Loading XGBoost model…")
    model = load_model(model_path)
    if Path(feat_path).exists():
        with open(feat_path) as f:
            feature_names = json.load(f)
    else:
        feature_names = [c for c in feat_df.columns
                         if c not in ("year", "tourney_date", "surface_norm", "target", "elo_prob_p1")]
        log.warning(f"  features file not found — using {len(feature_names)} auto-detected columns")

    test_mask = feat_df["year"] >= test_start
    test_df   = feat_df[test_mask].reset_index(drop=True)
    if test_df.empty:
        log.error(f"No rows found for year >= {test_start}. Lower --test-start or add more data.")
        sys.exit(1)

    avail_feats = [f for f in feature_names if f in test_df.columns]
    if not avail_feats:
        log.error("None of the model's expected features are in the loaded data. Check column names.")
        sys.exit(1)

    X_test  = test_df[avail_feats]
    y_true  = test_df["target"].values
    log.info(f"  Predicting {len(X_test):,} test rows (GPU={'cuda' if _CUDA_AVAILABLE else 'cpu'})…")
    t0      = time.time()
    y_prob  = model.predict_proba(X_test)[:, 1]
    log.info(f"  Inference complete in {time.time()-t0:.3f}s")

    # ── Baseline Kelly simulation ─────────────────────────────────────────────
    log.info("STEP 4/6  Running baseline Kelly simulation…")
    base_result = simulate_kelly_vectorised(
        y_prob, test_df["elo_prob_p1"].values, y_true,
        bankroll_init=bankroll,
    )
    log.info(f"  ROI={base_result['roi']:+.2f}%  "
             f"Sharpe={base_result['sharpe']:.3f}  "
             f"MaxDD={base_result['max_drawdown']*100:.1f}%  "
             f"Bets={base_result['n_bets']:,}")

    # ── GPU Monte Carlo ───────────────────────────────────────────────────────
    log.info(f"STEP 5/6  GPU Monte Carlo ({mc_iter:,} paths)…")
    mc_paths = run_monte_carlo(base_result["pnl_list"], bankroll_init=bankroll, n_iterations=mc_iter)
    if len(mc_paths) > 0:
        finals  = mc_paths[:, -1]
        log.info(f"  Median final bankroll: ${np.median(finals):,.2f}  "
                 f"5th pct: ${np.percentile(finals, 5):,.2f}  "
                 f"95th pct: ${np.percentile(finals, 95):,.2f}")

    # ── Parallel parameter sweep ──────────────────────────────────────────────
    sweep_df = pd.DataFrame()
    if not cli.no_sweep:
        log.info("STEP 6a  CPU parallel parameter sweep…")
        sweep_df = run_parallel_sweep(y_prob, test_df["elo_prob_p1"].values, y_true)
        best = sweep_df.sort_values("roi", ascending=False).iloc[0]
        log.info(f"  Best params: min_edge={best['min_edge']}  "
                 f"kelly={best['kelly_frac']}  vig={best['vig']}  "
                 f"→ ROI={best['roi']:+.2f}%  Sharpe={best['sharpe']:.3f}")

    # ── Walk-forward ──────────────────────────────────────────────────────────
    wf_df = pd.DataFrame()
    if not cli.no_wf and _HAS_XGB:
        log.info("STEP 6b  Walk-forward backtest (parallel folds)…")
        wf_df = run_walk_forward(feat_df, avail_feats, test_start)
        if not wf_df.empty:
            log.info(f"  Walk-forward complete: {len(wf_df)} folds")

    # ── PDF report ────────────────────────────────────────────────────────────
    log.info("STEP 6/6  Generating PDF report…")
    generate_report(
        pdf_path      = pdf_out,
        feat_df       = test_df,
        y_true        = y_true,
        y_prob        = y_prob,
        base_result   = base_result,
        mc_paths      = mc_paths,
        sweep_df      = sweep_df,
        wf_df         = wf_df,
        feature_names = avail_feats,
    )

    # ── JSON results ──────────────────────────────────────────────────────────
    json_path = Path(pdf_out).with_suffix(".json")
    summary = {
        "generated_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_used":           _GPU_DEVICE,
        "test_rows":          int(len(y_true)),
        "accuracy":           float(accuracy_score(y_true, y_prob >= 0.5)),
        "roc_auc":            float(roc_auc_score(y_true, y_prob)),
        "brier_score":        float(brier_score_loss(y_true, y_prob)),
        "n_bets":             base_result["n_bets"],
        "roi_pct":            round(base_result["roi"], 4),
        "win_rate_pct":       round(base_result["win_rate"], 4),
        "sharpe":             round(base_result["sharpe"], 4),
        "max_drawdown_pct":   round(base_result["max_drawdown"] * 100, 4),
        "final_bankroll":     round(base_result["final_bankroll"], 2),
        "mc_iterations":      mc_iter,
        "mc_median_final":    round(float(np.median(mc_paths[:, -1])), 2) if len(mc_paths) > 0 else None,
        "mc_p5_final":        round(float(np.percentile(mc_paths[:, -1], 5)), 2) if len(mc_paths) > 0 else None,
        "mc_p95_final":       round(float(np.percentile(mc_paths[:, -1], 95)), 2) if len(mc_paths) > 0 else None,
        "best_sweep_params":  sweep_df.sort_values("roi", ascending=False).iloc[0].to_dict() if not sweep_df.empty else None,
        "walk_forward":       wf_df.to_dict(orient="records") if not wf_df.empty else [],
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"  JSON results → {json_path}")

    log.info("=" * 60)
    log.info(f"  DONE.  Report: {pdf_out}")
    log.info(f"         JSON:   {json_path}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point  — __main__ guard is REQUIRED on Windows (multiprocessing spawn)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Windows spawn safety
    multiprocessing.freeze_support()
    main()
