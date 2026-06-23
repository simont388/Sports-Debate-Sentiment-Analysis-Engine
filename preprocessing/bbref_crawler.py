import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import time
import os

# --- Config ---

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; sports-debate-research/1.0)"
}

SLEEP_BETWEEN_REQUESTS = 4  # seconds — be polite, don't get blocked

OUTPUT_DIR = "data/stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Players in your 10 debates
# Verify IDs by visiting basketball-reference.com, searching the player,
# and checking the URL e.g. /players/j/jamesle01.html -> ID is jamesle01
PLAYERS = {
    "LeBron James":      "jamesle01",
    "Stephen Curry":     "curryst01",
    "Giannis":           "antetgi01",
    "Kobe Bryant":       "bryanko01",
    "Kawhi Leonard":     "leonaka01",
    "Russell Westbrook": "westbru01",
    "Carmelo Anthony":   "anthoca01",
    "Draymond Green":    "greendr01",
    "Nikola Jokic":      "jokicni01",
}

TEAMS = {
    "Golden State Warriors 2017": {
        "url": "https://www.basketball-reference.com/teams/GSW/2017.html",
        "filename": "warriors_2017_roster.csv"
    }
}


# --- HTML comment stripping ---

def extract_hidden_tables(html):
    """
    Basketball Reference hides many stat tables inside HTML comments
    to prevent easy scraping. This strips comment wrappers to expose them.
    """
    soup = BeautifulSoup(html, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment_soup = BeautifulSoup(comment, "html.parser")
        comment.replace_with(comment_soup)
    return soup


# --- Core fetch ---

def fetch_page(url):
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        print(f"  Fetched: {url}")
        return response.text
    except requests.RequestException as e:
        print(f"  Failed to fetch {url}: {e}")
        return None


# --- Row parser ---

def parse_table_rows(table, season_stat="year_id"):
    """
    Generic row parser. Skips mid-table header rows and empty rows.
    season_stat is the data-stat attribute used to identify valid rows.
    """
    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return rows
    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        th = tr.find(["th", "td"], {"data-stat": season_stat})
        if not th or not th.text.strip():
            continue
        rows.append(tr)
    return rows


def get_cell(tr, stat):
    td = tr.find(["td", "th"], {"data-stat": stat})
    return td.text.strip() if td else ""


# --- Traded player deduplication ---

def deduplicate_trades(df, season_col="Season", team_col="Team"):
    """
    BBRef adds a TOT row for traded players. Keep TOT, drop team-specific rows
    for that season so each season appears once.
    """
    if team_col not in df.columns:
        return df
    seasons_with_tot = df[df[team_col] == "TOT"][season_col].unique()
    mask = (df[season_col].isin(seasons_with_tot)) & (df[team_col] != "TOT")
    return df[~mask].reset_index(drop=True)


# --- Player stat fetchers ---

def get_per_game_stats(player_name, player_id):
    """
    Fetches season by season per game stats.
    Saves: data/stats/{player}_per_game.csv
    """
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"
    html = fetch_page(url)
    if not html:
        return None

    soup = extract_hidden_tables(html)
    table = soup.find("table", {"id": "per_game_stats"})
    if not table:
        print(f"  No per game table found for {player_name}")
        return None

    rows = []
    for tr in parse_table_rows(table, season_stat="year_id"):
        rows.append({
            "Season": get_cell(tr, "year_id"),
            "Age":    get_cell(tr, "age"),
            "Team":   get_cell(tr, "team_name_abbr"),
            "G":      get_cell(tr, "games"),
            "MP":     get_cell(tr, "mp_per_g"),
            "PPG":    get_cell(tr, "pts_per_g"),
            "RPG":    get_cell(tr, "trb_per_g"),
            "APG":    get_cell(tr, "ast_per_g"),
            "FG%":    get_cell(tr, "fg_pct"),
            "3P%":    get_cell(tr, "fg3_pct"),
            "FT%":    get_cell(tr, "ft_pct"),
        })

    if not rows:
        print(f"  No rows parsed for {player_name}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["PPG"] != ""]
    df = deduplicate_trades(df)

    filename = f"{player_name.lower().replace(' ', '_')}_per_game.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath} ({len(df)} seasons)")
    return df


def get_advanced_stats(player_name, player_id):
    """
    Fetches advanced stats per season.
    Saves: data/stats/{player}_advanced.csv
    """
    url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}.html"
    html = fetch_page(url)
    if not html:
        return None

    soup = extract_hidden_tables(html)
    table = soup.find("table", {"id": "advanced"})
    if not table:
        print(f"  No advanced table found for {player_name}")
        return None

    # Run the same debug check on the first row to verify advanced stat names
    tbody = table.find("tbody")
    if tbody:
        first_valid = next(
            (tr for tr in tbody.find_all("tr")
             if not (tr.get("class") and "thead" in tr.get("class"))),
            None
        )
        if first_valid:
            stats_found = [c.get("data-stat") for c in first_valid.find_all(["th","td"])]

    rows = []
    for tr in parse_table_rows(table, season_stat="year_id"):
        rows.append({
            "Season": get_cell(tr, "year_id"),
            "Age":    get_cell(tr, "age"),
            "Team":   get_cell(tr, "team_name_abbr"),
            "PER":    get_cell(tr, "per"),
            "TS%":    get_cell(tr, "ts_pct"),
            "BPM":    get_cell(tr, "bpm"),
            "VORP":   get_cell(tr, "vorp"),
            "WS":     get_cell(tr, "ws"),
            "WS/48":  get_cell(tr, "ws_per_48"),
        })

    if not rows:
        print(f"  No rows parsed for {player_name}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["PER"] != ""]
    df = deduplicate_trades(df)

    filename = f"{player_name.lower().replace(' ', '_')}_advanced.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath} ({len(df)} seasons)")
    return df


def get_team_roster_stats(team_name, url, filename):
    """
    Fetches per game stats for every player on a specific team season.
    Useful for the 2017 Warriors GOAT team claim.
    """
    html = fetch_page(url)
    if not html:
        return None

    soup = extract_hidden_tables(html)

    # Team pages may still use per_game — try both IDs
    table = soup.find("table", {"id": "per_game"})
    if not table:
        table = soup.find("table", {"id": "per_game_stats"})
    if not table:
        print(f"  No roster table found for {team_name}")
        return None

    rows = []
    tbody = table.find("tbody")
    if not tbody:
        return None

    for tr in tbody.find_all("tr"):
        if tr.get("class") and "thead" in tr.get("class"):
            continue
        # Team pages use name_display instead of player
        player_td = tr.find("td", {"data-stat": "name_display"})
        if not player_td or not player_td.text.strip():
            continue
        rows.append({
            "Player": player_td.text.strip(),
            "Age":    get_cell(tr, "age"),
            "G":      get_cell(tr, "games"),
            "PPG":    get_cell(tr, "pts_per_g"),
            "RPG":    get_cell(tr, "trb_per_g"),
            "APG":    get_cell(tr, "ast_per_g"),
            "FG%":    get_cell(tr, "fg_pct"),
            "3P%":    get_cell(tr, "fg3_pct"),
        })

    if not rows:
        print(f"  No rows parsed for {team_name}")
        return None

    df = pd.DataFrame(rows)
    df = df[df["PPG"] != ""]

    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath} ({len(df)} players)")
    return df


# --- Main runner ---

def run():
    print("=== Basketball Reference Crawler ===\n")

    print("--- Per Game Stats ---")
    for name, pid in PLAYERS.items():
        print(f"\n{name}:")
        get_per_game_stats(name, pid)

    print("\n--- Advanced Stats ---")
    for name, pid in PLAYERS.items():
        print(f"\n{name}:")
        get_advanced_stats(name, pid)

    print("\n--- Team Stats ---")
    for team_name, config in TEAMS.items():
        print(f"\n{team_name}:")
        get_team_roster_stats(team_name, config["url"], config["filename"])

    print("\n=== Done ===")
    print(f"All files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()