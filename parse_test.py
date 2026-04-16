import sys
from bs4 import BeautifulSoup

def main():
    with open('monte_carlo_results.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Tables of results are usually grouped by rounds in 'div.table-results-container' or something similar
    # In ATP Tour, it's usually table elements. Let's look for "Day" or "table"
    tables = soup.find_all('table', class_='day-table')
    if not tables:
        tables = soup.find_all('table', class_='scores-table')
    
    if not tables:
        # try simple tr/td approach
        for row in soup.select('tr'):
            players = row.select('td.day-table-name a')
            if len(players) == 2:
                # Top is winner, bottom is loser? Usually we can check which has the 'winner' class, or the first one is the winner in the HTML maybe
                print(f"Match: {players[0].text.strip()} vs {players[1].text.strip()}")
            elif len(players) > 0:
                print("Found row with players:", [p.text.strip() for p in players])

if __name__ == "__main__":
    main()
