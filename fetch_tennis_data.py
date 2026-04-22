import os
import requests
import zipfile
import io
import time

def download_file(url, output_path):
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {output_path}")
        else:
            print(f"File not found or error ({response.status_code}) for {url}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def download_and_extract_zip(url, output_dir):
    print(f"Downloading and extracting {url}...")
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(output_dir)
            print(f"Extracted zip contents to {output_dir}")
        else:
            print(f"File not found or error ({response.status_code}) for {url}")
    except Exception as e:
        print(f"Failed to process {url}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "ultimate_tennis_dataset")
    os.makedirs(data_dir, exist_ok=True)
    
    print("=========================================")
    print("  ULTIMATE TENNIS DATA DOWNLOADER        ")
    print("=========================================")
    print(f"Target directory: {data_dir}\n")

    # ==========================================
    # 1. MEN'S TENNIS (ATP, CHALLENGER, FUTURES)
    # ==========================================
    print("\n--- Fetching Jeff Sackmann MEN'S Data ---")
    atp_dir = os.path.join(data_dir, "mens_tour")
    os.makedirs(atp_dir, exist_ok=True)
    
    for year in range(1968, 2025):
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv", os.path.join(atp_dir, f"atp_matches_{year}.csv"))
    for year in range(1978, 2025):
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_qual_chall_{year}.csv", os.path.join(atp_dir, f"atp_matches_qual_chall_{year}.csv"))
    for year in range(1991, 2025):
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_futures_{year}.csv", os.path.join(atp_dir, f"atp_matches_futures_{year}.csv"))
        
    download_file("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv", os.path.join(atp_dir, "atp_players.csv"))
    for decade in ['70s', '80s', '90s', '00s', '10s', '20s', 'current']:
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_rankings_{decade}.csv", os.path.join(atp_dir, f"atp_rankings_{decade}.csv"))

    # ==========================================
    # 2. WOMEN'S TENNIS (WTA, ITF)
    # ==========================================
    print("\n--- Fetching Jeff Sackmann WOMEN'S Data ---")
    wta_dir = os.path.join(data_dir, "womens_tour")
    os.makedirs(wta_dir, exist_ok=True)
    
    for year in range(1920, 2025):
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv", os.path.join(wta_dir, f"wta_matches_{year}.csv"))
    for year in range(1994, 2025):
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_qual_itf_{year}.csv", os.path.join(wta_dir, f"wta_matches_qual_itf_{year}.csv"))
        
    download_file("https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv", os.path.join(wta_dir, "wta_players.csv"))
    for decade in ['80s', '90s', '00s', '10s', '20s', 'current']:
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_rankings_{decade}.csv", os.path.join(wta_dir, f"wta_rankings_{decade}.csv"))

    # ==========================================
    # 3. GRAND SLAM POINT-BY-POINT DATA
    # ==========================================
    print("\n--- Fetching Grand Slam Point-by-Point Data ---")
    pbp_dir = os.path.join(data_dir, "point_by_point")
    os.makedirs(pbp_dir, exist_ok=True)
    
    # Download meta and generic files
    for pbp_file in ['pbp_matches_atp_main.csv', 'pbp_matches_wta_main.csv']:
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/{pbp_file}", os.path.join(pbp_dir, pbp_file))
        
    for year in range(2011, 2025):
        for slam in ['ausopen', 'frenchopen', 'wimbledon', 'usopen']:
            url = f"https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/{year}-{slam}-points.csv"
            download_file(url, os.path.join(pbp_dir, f"{year}-{slam}-points.csv"))
            url_matches = f"https://raw.githubusercontent.com/JeffSackmann/tennis_slam_pointbypoint/master/{year}-{slam}-matches.csv"
            download_file(url_matches, os.path.join(pbp_dir, f"{year}-{slam}-matches.csv"))

    # ==========================================
    # 4. MATCH CHARTING PROJECT (High Granularity)
    # ==========================================
    print("\n--- Fetching Match Charting Project Data ---")
    charting_dir = os.path.join(data_dir, "match_charting")
    os.makedirs(charting_dir, exist_ok=True)
    
    charting_files = [
        "charting-m-matches.csv", "charting-m-points.csv", "charting-m-stats-ReturnDepth.csv", 
        "charting-m-stats-ReturnOutcomes.csv", "charting-m-stats-ShotDirection.csv", "charting-m-stats-ShotTypes.csv",
        "charting-w-matches.csv", "charting-w-points.csv", "charting-w-stats-ReturnDepth.csv",
        "charting-w-stats-ReturnOutcomes.csv", "charting-w-stats-ShotDirection.csv", "charting-w-stats-ShotTypes.csv"
    ]
    for c_file in charting_files:
        download_file(f"https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/{c_file}", os.path.join(charting_dir, c_file))

    # ==========================================
    # 5. TENNIS-DATA.CO.UK BETTING ODDS
    # ==========================================
    print("\n--- Fetching Tennis-Data.co.uk Historical Odds Data ---")
    odds_dir = os.path.join(data_dir, "betting_odds")
    os.makedirs(odds_dir, exist_ok=True)
    
    # ATP Odds
    atp_odds_dir = os.path.join(odds_dir, "ATP")
    os.makedirs(atp_odds_dir, exist_ok=True)
    for year in range(2001, 2025):
        url = f"http://www.tennis-data.co.uk/{year}/{year}.zip"
        download_and_extract_zip(url, atp_odds_dir)
        
    # WTA Odds
    wta_odds_dir = os.path.join(odds_dir, "WTA")
    os.makedirs(wta_odds_dir, exist_ok=True)
    for year in range(2007, 2025):
        url = f"http://www.tennis-data.co.uk/{year}w/{year}w.zip"
        download_and_extract_zip(url, wta_odds_dir)

    print("\n=========================================")
    print(f"Download complete! Ultimate dataset saved in:\n{data_dir}")
    print("=========================================")

if __name__ == '__main__':
    main()
