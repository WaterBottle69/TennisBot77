"""
ingest.py — Data ingestion for the tennis mentality score backtest pipeline.

Downloads and caches ATP match CSVs, player data, and SLAM PBP files
from JeffSackmann's tennis_atp GitHub repository.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SKIPPED: list = []


def get_skipped() -> list:
    return list(_SKIPPED)


ATP_MATCHES_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
)
ATP_PLAYERS_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
)
SLAM_PBP_URL = (
    "https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/"
    "{year}-{slam}-matches.csv"
)

DEFAULT_CACHE_DIR = Path(__file__).parent / ".cache"


def download_file(url: str, cache_dir: Path) -> Optional[Path]:
    """Download a URL to a cache directory using an md5-hashed filename.

    Returns the cached Path on success, or None on any failure.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    url_hash = hashlib.md5(url.encode()).hexdigest()
    cached_path = cache_dir / url_hash

    if cached_path.exists() and cached_path.stat().st_size > 0:
        return cached_path

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            msg = f"{url} → HTTP {resp.status_code}"
            logger.warning("HTTP %s for %s — skipping", resp.status_code, url)
            _SKIPPED.append(msg)
            return None
        cached_path.write_bytes(resp.content)
        return cached_path
    except Exception as exc:
        msg = f"{url} → {exc}"
        logger.warning("Failed to download %s: %s — skipping", url, exc)
        _SKIPPED.append(msg)
        return None


def load_atp_matches(
    years: List[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download and concatenate ATP match CSVs for the requested years.

    tourney_date is parsed as YYYYMMDD string → pd.Timestamp.
    Rows with null scores or walkovers ('W/O') are dropped.
    Retirements ('RET') are kept.
    """
    frames = []
    for year in years:
        url = ATP_MATCHES_URL.format(year=year)
        path = download_file(url, cache_dir)
        if path is None:
            logger.warning("Skipping ATP matches for year %s", year)
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            # Parse tourney_date
            df["tourney_date"] = pd.to_datetime(
                df["tourney_date"].astype(str).str.zfill(8), format="%Y%m%d", errors="coerce"
            )
            # Drop null scores and walkovers
            df = df[df["score"].notna()]
            df = df[~df["score"].astype(str).str.contains(r"W/O", na=False)]
            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to parse ATP matches for year %s: %s", year, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    return combined


def load_atp_players(cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Download the ATP player database CSV."""
    path = download_file(ATP_PLAYERS_URL, cache_dir)
    if path is None:
        logger.warning("Could not load ATP players CSV")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except Exception as exc:
        logger.warning("Failed to parse ATP players CSV: %s", exc)
        return pd.DataFrame()


def load_slam_pbp(
    years: List[int],
    slams: List[str],
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download SLAM point-by-point files for the requested years and slams.

    Silently skips any file that returns 404 or fails to download/parse.
    If challenge columns are absent, sets them to NaN and continues.
    """
    frames = []
    for year in years:
        for slam in slams:
            url = SLAM_PBP_URL.format(year=year, slam=slam)
            path = download_file(url, cache_dir)
            if path is None:
                continue
            try:
                df = pd.read_csv(path, low_memory=False)
                # Check for challenge columns
                challenge_cols = [
                    c for c in df.columns if "challenge" in c.lower() or "Challenge" in c
                ]
                if not challenge_cols:
                    # No challenge data — add NaN placeholders
                    df["p1_challenge_success"] = float("nan")
                    df["p2_challenge_success"] = float("nan")
                    df["p1_challenges_total"] = float("nan")
                    df["p2_challenges_total"] = float("nan")
                df["year"] = year
                df["slam"] = slam
                frames.append(df)
            except Exception as exc:
                logger.warning(
                    "Failed to parse SLAM PBP for %s-%s: %s", year, slam, exc
                )

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
