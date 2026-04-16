import asyncio
import csv
import io
import json
import os
import sys

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tennis_scraper import TennisStatsScraper

RAW_CSV = """round,winner_name,loser_name,score
Final,Jannik Sinner,Carlos Alcaraz,7-6(5) 6-3
Semi-Finals,Carlos Alcaraz,Valentin Vacherot,6-4 6-4
Semi-Finals,Jannik Sinner,Alexander Zverev,6-1 6-4
Quarter-Finals,Carlos Alcaraz,Alexander Bublik,6-3 6-0
Quarter-Finals,Jannik Sinner,Felix Auger-Aliassime,6-3 6-4
Quarter-Finals,Alexander Zverev,Joao Fonseca,7-5 6-7(3) 6-3
Quarter-Finals,Valentin Vacherot,Alex de Minaur,6-4 3-6 6-3
Round of 16,Carlos Alcaraz,Tomas Martin Etcheverry,6-1 4-6 6-3
Round of 16,Jannik Sinner,Tomas Machac,6-1 6-7(3) 6-3
Round of 16,Alexander Zverev,Zizou Bergs,6-2 7-5
Round of 16,Valentin Vacherot,Hubert Hurkacz,6-7(4) 6-3 6-4
Round of 16,Alex de Minaur,Alexander Blockx,7-5 7-6(4)
Round of 16,Felix Auger-Aliassime,Casper Ruud,7-5 2-2 RET
Round of 16,Alexander Bublik,Jiri Lehecka,6-2 7-5
Round of 16,Joao Fonseca,Matteo Berrettini,6-3 6-2
Round of 32,Carlos Alcaraz,Sebastian Baez,6-1 6-3
Round of 32,Jannik Sinner,Ugo Humbert,6-3 6-0
Round of 32,Alexander Zverev,Cristian Garin,4-6 6-4 7-5
Round of 32,Casper Ruud,Corentin Moutet,7-5 6-3
Round of 32,Alex de Minaur,Cameron Norrie,7-6(5) 2-6 6-2
Round of 32,Felix Auger-Aliassime,Marin Cilic,7-6(4) 6-3
Round of 32,Alexander Bublik,Gael Monfils,6-4 6-4
Round of 32,Hubert Hurkacz,Fabian Marozsan,6-2 6-3
Round of 32,Tomas Machac,Francisco Cerundolo,7-6(2) 6-3
Round of 32,Tomas Martin Etcheverry,Terence Atmane,3-6 6-3 6-2
Round of 32,Joao Fonseca,Arthur Rinderknech,7-5 4-6 6-3
Round of 32,Valentin Vacherot,Lorenzo Musetti,7-6(6) 7-5
Round of 32,Matteo Berrettini,Daniil Medvedev,6-0 6-0
Round of 32,Alexander Blockx,Flavio Cobolli,6-3 6-3
Round of 32,Jiri Lehecka,Alejandro Tabilo,4-6 7-6(4) 6-3
Round of 32,Zizou Bergs,Andrey Rublev,6-4 6-1
Round of 64,Casper Ruud,Alexei Popyrin,6-3 6-4
Round of 64,Hubert Hurkacz,Luciano Darderi,7-6(4) 5-7 6-1
Round of 64,Terence Atmane,Ethan Quinn,6-1 6-4
Round of 64,Matteo Berrettini,Roberto Bautista Agut,4-0 RET
Round of 64,Marin Cilic,Alexander Shevchenko,6-1 6-3
Round of 64,Tomas Martin Etcheverry,Grigor Dimitrov,6-4 2-6 6-3
Round of 64,Fabian Marozsan,Damir Dzumhur,6-2 6-1
Round of 64,Corentin Moutet,Alexandre Muller,6-4 6-1
Round of 64,Flavio Cobolli,Francisco Comesana,7-5 2-6 6-3
Round of 64,Jiri Lehecka,Emilio Nava,7-6(1) 6-7(8) 6-2
Round of 64,Arthur Rinderknech,Karen Khachanov,7-5 6-2
Round of 64,Andrey Rublev,Nuno Borges,6-4 1-6 6-1
Round of 64,Zizou Bergs,Adrian Mannarino,6-4 6-3
Round of 64,Alexander Blockx,Denis Shapovalov,6-4 4-6 6-3
Round of 64,Joao Fonseca,Gabriel Diallo,6-2 6-3
Round of 64,Cristian Garin,Matteo Arnaldi,6-2 6-4
Round of 64,Tomas Machac,Daniel Altmaier,6-4 1-6 6-3
Round of 64,Valentin Vacherot,Juan Manuel Cerundolo,5-7 6-2 6-1
Round of 64,Francisco Cerundolo,Stefanos Tsitsipas,7-5 6-4
Round of 64,Sebastian Baez,Stan Wawrinka,7-5 7-5
Round of 64,Ugo Humbert,Moise Kouame,6-3 7-5
Round of 64,Gael Monfils,Tallon Griekspoor,6-7(7) 6-1 6-4
Round of 64,Cameron Norrie,Miomir Kecmanovic,6-2 4-6 7-6(0)
Round of 64,Alejandro Tabilo,Marton Fucsovics,6-4 6-3"""

async def main():
    # Load player id map
    with open('player_id_map.json', 'r') as f:
        player_map = json.load(f)
        
    scraper = TennisStatsScraper()
    
    rows = []
    # Use DictReader
    reader = csv.DictReader(io.StringIO(RAW_CSV))
    for i, row in enumerate(reader):
        w_name = row["winner_name"]
        l_name = row["loser_name"]
        
        # Scrape player details
        w_data = await scraper.get_player_data(w_name)
        l_data = await scraper.get_player_data(l_name)
        
        # Get Jeff Sackmann IDs
        w_id = player_map.get(w_name, 200000 + i) # Dummy IDs if not found to avoid crashing ELO
        l_id = player_map.get(l_name, 200000 + i + 100)
        
        # Construct row
        # Must match `backtester.py` logic which reads `winner_hand`, `winner_ht`, etc.
        out_row = {
            "tourney_id": "2026-0410",
            "tourney_name": "Monte Carlo Masters",
            "surface": "Clay",
            "best_of": 3,
            "tourney_date": 20260412,
            "match_num": 1000 - i, # descending so oldest matches could be processed first, but wait, the CSV from browser might be newest to oldest
            "winner_name": w_name,
            "winner_id": w_id,
            "winner_hand": w_data.get("hand", "R"),
            "winner_ht": w_data.get("height_cm", 185),
            "winner_age": w_data.get("age", 25),
            "winner_rank": w_data.get("ranking", 50),
            "winner_rank_points": w_data.get("elo", 1500), # using elo proxy or could be left blank
            "loser_name": l_name,
            "loser_id": l_id,
            "loser_hand": l_data.get("hand", "R"),
            "loser_ht": l_data.get("height_cm", 185),
            "loser_age": l_data.get("age", 25),
            "loser_rank": l_data.get("ranking", 50),
            "loser_rank_points": l_data.get("elo", 1500),
            "score": row["score"],
            "round": row["round"]
        }
        rows.append(out_row)

    # Reverse rows to chronological order (Round of 64 -> Final)
    rows.reverse()
    
    # Write to atp_matches_2026.csv
    out_dir = os.path.expanduser("~/Downloads/tennis_atp-master")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "atp_matches_2026.csv")
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
            
    print(f"Successfully generated {out_path} with {len(rows)} matches.")

if __name__ == "__main__":
    asyncio.run(main())
