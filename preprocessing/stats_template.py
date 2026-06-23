import os
import json
import pandas as pd

# Generate synthetic stat paragraphs from CSV data for inclusion in the corpus
# Handles per-game, advanced, trend, comparison, and team roster paragraphs

STATS_DIR = "data/stats"
OUTPUT_DIR = "data/corpus"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "stat_paragraphs.json")

# Define the players, their prime and recent seasons, and associated debates
PLAYER_WINDOWS = {
    "lebron_james": {
        "display": "LeBron James",
        "prime": ["2011-12", "2012-13", "2013-14", "2014-15", "2015-16", "2016-17", "2017-18"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "debates": ["lebron_washed", "kobe_vs_lebron_scoring"]
    },
    "stephen_curry": {
        "display": "Stephen Curry",
        "prime": ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "debates": ["curry_system_player", "warriors_2017_goat_team"]
    },
    "giannis": {
        "display": "Giannis Antetokounmpo",
        "prime": ["2018-19", "2019-20", "2020-21", "2021-22"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "relevant_seasons": ["2020-21", "2021-22", "2022-23"],
        "debates": ["giannis_no_costar"]
    },
    "kobe_bryant": {
        "display": "Kobe Bryant",
        "prime": ["2002-03", "2004-05", "2005-06", "2006-07", "2007-08", "2008-09", "2009-10"],
        "recent":   ["2012-13", "2013-14", "2014-15"],
        "debates": ["kobe_vs_lebron_scoring"]
    },
    "kawhi_leonard": {
        "display": "Kawhi Leonard",
        "prime": ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "debates": ["kawhi_overrated"]
    },
    "russell_westbrook": {
        "display": "Russell Westbrook",
        "prime": ["2015-16", "2016-17", "2017-18", "2018-19"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "relevant_seasons": ["2021-22", "2022-23"],
        "debates": ["westbrook_ruined_lakers"]
    },
    "carmelo_anthony": {
        "display": "Carmelo Anthony",
        "prime": ["2005-06", "2006-07", "2007-08", "2008-09", "2009-10", "2011-12", "2012-13"],
        "recent":   ["2019-20", "2020-21", "2021-22"],
        "relevant_seasons": ["2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16"],
        "debates": ["carmelo_selfish"]
    },
    "draymond_green": {
        "display": "Draymond Green",
        "prime": ["2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20"],
        "recent":   ["2022-23", "2023-24", "2024-25"],
        "debates": ["draymond_most_important_warrior"]
    },
    "nikola_jokic": {
        "display": "Nikola Jokic",
        "prime": ["2019-20", "2020-21", "2021-22", "2022-23"],
        "recent":   ["2023-24", "2024-25"],
        "debates": ["jokic_most_complete"]
    }
}
# Stats get the highest weight since they are objective evidence
SOURCE_WEIGHT = 1.0

def safe_float(value, default=None):
    # Convert a value safely to a float
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def trend_word(prime_avg, recent_avg, stat_name):
    # Determine if the trend is an improvement, decline, or stable based on the stat type
    higher_is_better = ['PPG', 'RPG', 'APG', 'FG%', '3P%', 'FT%', 'PER', 'TS%', 'BPM', 'VORP', 'WS', 'WS/48']
    lower_is_better = []   # Currently no stats for this but can be extended in the future
    if stat_name in higher_is_better:
        if recent_avg > prime_avg:
            return "improved"
        elif recent_avg < prime_avg:
            return "declined"
        else:
            return "remained stable"
    elif stat_name in lower_is_better:
        if recent_avg < prime_avg:
            return "improved"
        elif recent_avg > prime_avg:
            return "declined"
        else:
            return "remained stable"
    
    return "changed" # Default if we don't know the direction of improvement

def format_stat(val, stat_name):
    # Format percentages appropriately, and other stats with one decimal place
    pct_stats = ['FG%', '3P%', 'FT%', 'TS%']
    if stat_name in pct_stats:
        return f"{val * 100:.1f}%" if val < 1.0 else f"{val:.1f}%"
    return f"{val:.1f}"

def generate_season_paragraph(player_display, row):
    # Generate a paragraph summarizing the player's stats for a given season
    # First we extract the relevant stats, handling missing or non-numeric values gracefully
    season = row.get('Season', 'Unknown Season')
    team = row.get('Team', 'Unknown Team')
    ppg = safe_float(row.get('PPG'))
    rpg = safe_float(row.get('RPG'))
    apg = safe_float(row.get('APG'))
    fg_pct = safe_float(row.get('FG%'))
    three_pt_pct = safe_float(row.get('3P%'))
    ft_pct = safe_float(row.get('FT%'))
    games = row.get('G', '')

    # Build the basic stats part of the paragraph, only including available stats
    parts = []
    if ppg is not None:
        parts.append(f"{ppg:.1f} points")
    if rpg is not None:
        parts.append(f"{rpg:.1f} rebounds")
    if apg is not None:
        parts.append(f"{apg:.1f} assists")
    
    stat_str = ", ".join(parts) if parts else "statistics unavailable"

    # Build the shooting percentages part of the paragraph
    shoot_parts = []
    if fg_pct is not None:
        shoot_parts.append(f"{format_stat(fg_pct, 'FG%')} from the field")
    if three_pt_pct is not None:
        shoot_parts.append(f"{format_stat(three_pt_pct, '3P%')} from three point range")
    if ft_pct is not None:
        shoot_parts.append(f"{format_stat(ft_pct, 'FT%')} from the free throw line")
    shoot_str = " and ".join(shoot_parts) if shoot_parts else ""
    games_str = f" in {games} games" if games else ""

    para = (
        f"In the {season} season, {player_display} averaged {stat_str} per game for the team {team}{games_str}, "
        f"shooting {shoot_str}."
    )
    return para

def generate_advanced_season_paragraph(player_display, row):
    # Generate a paragraph summarizing the player's advanced stats for a given season
    # Extract advanced stats, handling missing or non-numeric values gracefully
    season = row.get('Season', 'Unknown Season')
    per = safe_float(row.get('PER'))
    ts_pct = safe_float(row.get('TS%'))
    bpm = safe_float(row.get('BPM'))
    vorp = safe_float(row.get('VORP'))
    ws = safe_float(row.get('WS'))
    ws_48 = safe_float(row.get('WS/48'))

    # Build the advanced stats part of the paragraph, only including available stats
    parts = []
    if per is not None:
        parts.append(f"A Player Efficiency Rating (PER) of {per:.1f}")
    if ts_pct is not None:
        parts.append(f"a True Shooting Percentage (TS%) of {format_stat(ts_pct, 'TS%')}")
    if bpm is not None:
        direction = "positive" if bpm >= 0 else "negative"
        parts.append(f"a Box Plus/Minus (BPM) of {bpm:.1f} ({direction} impact)")
    if vorp is not None:
        parts.append(f"a Value Over Replacement Player (VORP) of {vorp:.1f}")
    if ws is not None:
        parts.append(f"{ws:.1f} Win Shares (WS)")
    if ws_48 is not None:
        parts.append(f"{ws_48:.3f} Win Shares per 48 minutes (WS/48)")
    
    if not parts:
        return f"Advanced statistics for {player_display} in the season {season} are unavailable."
    
    # Combine the parts into a coherent paragraph
    stat_str = ", ".join(parts[:-1])
    # Add "and" before the last part if there are multiple stats
    if len(parts) > 1:
        stat_str += f", and {parts[-1]}"
    else:
        stat_str = parts[0]
    return f"In the season {season}, {player_display} posted {stat_str}."

def generate_trend_paragraph(player_display, df, stat, prime_seasons, recent_seasons):
    # Generate a paragraph analyzing the trend of a specific stat from the player's prime seasons to recent seasons
    prime_rows = df[df['Season'].isin(prime_seasons)]
    recent_rows = df[df['Season'].isin(recent_seasons)]

    prime_vals = [safe_float(val) for val in prime_rows[stat] if safe_float(val) is not None]
    recent_vals = [safe_float(val) for val in recent_rows[stat] if safe_float(val) is not None]

    if not prime_vals or not recent_vals:
        return f"Not enough data to analyze the trend of {stat} for {player_display}."
    
    # Calculate averages and determine the direction of the trend
    prime_avg = sum(prime_vals) / len(prime_vals)
    recent_avg = sum(recent_vals) / len(recent_vals)
    delta = abs(prime_avg - recent_avg)
    direction = trend_word(prime_avg, recent_avg, stat)

    prime_range = f"{min(prime_seasons)[:4]}-{max(prime_seasons)[:4]}"
    recent_range = f"{min(recent_seasons)[:4]}-{max(recent_seasons)[:4]}"
    # Map stat abbreviations to more descriptive labels for the paragraph
    stat_labels = {
        "PPG": "scoring average",
        "RPG":   "rebounding average",
        "APG":   "assists average",
        "FG%":   "field goal percentage",
        "3P%":   "three point percentage",
        "FT%":   "free throw percentage",
        "PER":   "Player Efficiency Rating",
        "TS%":   "true shooting percentage",
        "BPM":   "Box Plus Minus",
        "VORP":  "Value Over Replacement Player",
        "WS":    "Win Shares",
        "WS/48": "Win Shares per 48 minutes",
    }

    stat_label = stat_labels.get(stat, stat)
    # Format the stats for display in the paragraph
    prime_fmt = format_stat(prime_avg, stat)
    recent_fmt = format_stat(recent_avg, stat)
    delta_fmt = format_stat(delta, stat)

    return(
        f"{player_display}'s {stat_label} has {direction} from an average of {prime_fmt} "
        f"during their prime seasons ({prime_range}) to {recent_fmt} in recent seasons "
        f"({recent_range}), a difference of {delta_fmt}."
    )

def generate_comparison_paragraph(player_a, player_b, df_a, df_b, stat, season):
    # Generate a paragraph comparing a specific stat between two players for a given season
    row_a = df_a[df_a['Season'] == season]
    row_b = df_b[df_b['Season'] == season]
    if row_a.empty or row_b.empty:
        return f"Data for the season {season} is unavailable for comparison between {player_a} and {player_b}."
    
    val_a = safe_float(row_a.iloc[0].get(stat))
    val_b = safe_float(row_b.iloc[0].get(stat))
    if val_a is None or val_b is None:
        return f"{stat} data for the season {season} is unavailable for comparison between {player_a} and {player_b}."
    
    stat_labels = {
        "PPG": "points per game",
        "RPG": "rebounds per game",
        "APG": "assists per game",
        "FG%": "field goal percentage",
        "3P%": "three point percentage",
        "PER": "Player Efficiency Rating",
        "TS%": "true shooting percentage",
        "BPM": "Box Plus Minus",
    }
    stat_label = stat_labels.get(stat, stat)
    # Determine which player had the better stat for that season
    higher = player_a if val_a > val_b else player_b
    fmt_a = format_stat(val_a, stat)
    fmt_b = format_stat(val_b, stat)

    return(
        f"In the season {season}, {player_a} averaged {fmt_a} {stat_label} "
        f"comapred to {player_b}'s {fmt_b}, making {higher} the better performer in this category that year."
    )

def build_stat_paragraphs():
    # Main function to build the stat paragraphs for all players and save to a JSON file
    all_paragraphs = []
    
    for player_key, config in PLAYER_WINDOWS.items():
        # Extract configuration for the player
        display = config["display"]
        prime_seasons = config["prime"]
        recent_seasons = config["recent"]
        debates = config["debates"]
        # Construct file paths for per game and advanced stats
        pg_path = os.path.join(STATS_DIR, f"{player_key}_per_game.csv")
        adv_path = os.path.join(STATS_DIR, f"{player_key}_advanced.csv")
        # Process per game stats if the file exists
        if os.path.exists(pg_path):
            df_pg = pd.read_csv(pg_path)
            print(f"Processing per game stats: {player_key} ({len(df_pg)} seasons)")
            for _, row in df_pg.iterrows():
                para = generate_season_paragraph(display, row)
                if para:
                    # Add the generated paragraph to the list with metadata about the source, player, season, and associated debates
                    all_paragraphs.append({
                        "text": para,
                        "source": "Basketball Reference",
                        "source_weight": SOURCE_WEIGHT,
                        "player": display,
                        "season": row.get('Season', ''),
                        "paragraph_type": "season_stats",
                        "debates": debates,
                        "argument_side": "neutral"
                    })
            # Generate trend comparison paragraphs for key stats
            for stat in ['PPG', 'RPG', 'APG', 'FG%', '3P%']:
                if stat in df_pg.columns:
                    para = generate_trend_paragraph(display, df_pg, stat, prime_seasons, recent_seasons)
                    if para:
                        prime_vals = [safe_float(val) for val in 
                                      df_pg[df_pg['Season'].isin(prime_seasons)][stat] 
                                      if safe_float(val) is not None]
                        recent_vals = [safe_float(val) for val in 
                                       df_pg[df_pg['Season'].isin(recent_seasons)][stat] 
                                       if safe_float(val) is not None]
                        if prime_vals and recent_vals:
                            prime_avg = sum(prime_vals) / len(prime_vals)
                            recent_avg = sum(recent_vals) / len(recent_vals)
                            side = "supporting" if recent_avg < prime_avg else "opposing"
                        else:
                            side = "neutral"
                        # Add the trend comparison paragraph to the list with metadata about the source, player, season, stat, and associated debates
                        all_paragraphs.append({
                            "text":             para,
                            "source":           "Basketball Reference",
                            "source_weight":    SOURCE_WEIGHT,
                            "player":           display,
                            "season":           "career",
                            "paragraph_type":   "trend_comparison",
                            "stat":             stat,
                            "debates":          debates,
                            "argument_side":    side
                        })
        else:
            print(f"Warning: Per game stats file not found for {player_key} at {pg_path}")
        # Process advanced stats if the file exists
        if os.path.exists(adv_path):
            df_adv = pd.read_csv(adv_path)
            print(f"Processing advanced stats: {player_key} ({len(df_adv)} seasons)")

            for _, row in df_adv.iterrows():
                para = generate_advanced_season_paragraph(display, row)
                if para:
                    # Add the generated advanced stats paragraph to the list with metadata about the source, player, season, and associated debates
                    all_paragraphs.append({
                        "text":             para,
                        "source":           "Basketball Reference",
                        "source_weight":    SOURCE_WEIGHT,
                        "player":           display,
                        "season":           row.get("Season", ""),
                        "paragraph_type":   "advanced_stats",
                        "debates":          debates,
                        "argument_side":    "neutral"
                    })
            # Generate trend comparison paragraphs for key advanced stats
            for stat in ['PER', 'TS%', 'BPM', 'VORP', 'WS', 'WS/48']:
                if stat in df_adv.columns:
                    para = generate_trend_paragraph(display, df_adv, stat, prime_seasons, recent_seasons)
                    if para:
                        prime_vals = [safe_float(val) for val in 
                                      df_adv[df_adv['Season'].isin(prime_seasons)][stat] 
                                      if safe_float(val) is not None]
                        recent_vals = [safe_float(val) for val in 
                                       df_adv[df_adv['Season'].isin(recent_seasons)][stat] 
                                       if safe_float(val) is not None]
                        if prime_vals and recent_vals:
                            prime_avg = sum(prime_vals) / len(prime_vals)
                            recent_avg = sum(recent_vals) / len(recent_vals)
                            side = "supporting" if recent_avg < prime_avg else "opposing"
                        else:
                            side = "neutral"
                        # Add the advanced stat trend comparison paragraph to the list with metadata about the source, player, season, stat, and associated debates
                        all_paragraphs.append({
                            "text":             para,
                            "source":           "Basketball Reference",
                            "source_weight":    SOURCE_WEIGHT,
                            "player":           display,
                            "season":           "career",
                            "paragraph_type":   "trend_comparison",
                            "stat":             stat,
                            "debates":          debates,
                            "argument_side":    side
                        })
        else:
            print(f"Warning: Advanced stats file not found for {player_key} at {adv_path}")
    
    # Generate head-to-head comparison paragraphs for Kobe Bryant vs LeBron James for seasons where they overlapped
    # Works for the Kobe bs Lebron scoring debate, could be extended if more player vs player debates added
    kobe_pg_path = os.path.join(STATS_DIR, "kobe_bryant_per_game.csv")
    lebron_pg_path = os.path.join(STATS_DIR, "lebron_james_per_game.csv")
    if os.path.exists(kobe_pg_path) and os.path.exists(lebron_pg_path):
        # Load stats for each player and find overlapping seasons to compare head-to-head
        df_kobe = pd.read_csv(kobe_pg_path)
        df_lebron = pd.read_csv(lebron_pg_path)
        kobe_seasons = set(df_kobe['Season'].tolist())
        lebron_seasons = set(df_lebron['Season'].tolist())
        overlap = kobe_seasons.intersection(lebron_seasons)

        print(f"Generating comparison paragraphs for Kobe vs LeBron in seasons: {overlap}")
        for season in sorted(overlap):
            for stat in ["PPG", "FG%", "3P%", "TS%"]:
                # For TS% we want to use the advanced stats since it is more accurate
                if stat == "TS%":
                    kobe_adv_path = os.path.join(STATS_DIR, "kobe_bryant_advanced.csv")
                    lebron_adv_path = os.path.join(STATS_DIR, "lebron_james_advanced.csv")
                    if os.path.exists(kobe_adv_path) and os.path.exists(lebron_adv_path):
                        df_kobe_adv = pd.read_csv(kobe_adv_path)
                        df_lebron_adv = pd.read_csv(lebron_adv_path)
                        para = generate_comparison_paragraph("Kobe Bryant", "Lebron James", df_kobe_adv, df_lebron_adv, stat, season)
                else:
                    # For other stats we can use the per game stats for the head-to-head comparison
                    para = generate_comparison_paragraph("Kobe Bryant", "Lebron James", df_kobe, df_lebron, stat, season)
                if para:
                    # Add the head-to-head comparison paragraph to the list with metadata about the source, players, season, stat, and associated debates
                    all_paragraphs.append({
                        "text":             para,
                        "source":           "Basketball Reference",
                        "source_weight":    SOURCE_WEIGHT,
                        "player":           "Kobe Bryant vs LeBron James",
                        "season":           season,
                        "paragraph_type":   "head_to_head_comparison",
                        "stat":             stat,
                        "debates":          ["kobe_vs_lebron_scoring"],
                        "argument_side":    "neutral"
                    })

    # Generate paragraphs summarizing stat for Warrioris 2017 roster, for debate about best team of all time
    # Can be extended if similar debates added
    warriors_path = os.path.join(STATS_DIR, "warriors_2017_roster.csv")
    if os.path.exists(warriors_path):
        df_warriors = pd.read_csv(warriors_path)
        print(f"Generating Warrioirs 2017 team paragraph")
        # Iterate through each player on the roster
        for _, row in df_warriors.iterrows():
            player = row.get("Player", "Unknown Player")
            ppg = safe_float(row.get("PPG"))
            rpg = safe_float(row.get("RPG"))
            apg = safe_float(row.get("APG"))
            fg_pct = safe_float(row.get("FG%"))
            three_pt_pct = safe_float(row.get("3P%"))

            if not player or ppg is None:
                continue
            # Build the basic stats part of the paragraph, only including available stats
            parts = []
            if ppg is not None:
                parts.append(f"{ppg:.1f} points")
            if rpg is not None:
                parts.append(f"{rpg:.1f} rebounds")
            if apg is not None:
                parts.append(f"{apg:.1f} assists")
            stat_str = ", ".join(parts) if parts else "statistics unavailable"
            # Build the shooting percentages part of the paragraph
            shoot_parts = []
            if fg_pct is not None:
                shoot_parts.append(f"{format_stat(fg_pct, 'FG%')} from the field")
            if three_pt_pct is not None:
                shoot_parts.append(f"{format_stat(three_pt_pct, '3P%')} from three point range")
            shoot_str = " and ".join(shoot_parts) if shoot_parts else ""
            para = (
                f"On the 2016-17 Golden State Warriors, {player} averaged {stat_str} per game, "
                f"shooting {shoot_str}."
            )
            # Add the Warriors 2017 team roster paragraph to the list with metadata about the source, player, season, and associated debates
            all_paragraphs.append({
                "text":             para,
                "source":           "Basketball Reference",
                "source_weight":    SOURCE_WEIGHT,
                "player":           player,
                "season":           "2016-17",
                "paragraph_type":   "team_roster_stats",
                "debates":          ["warriors_2017_goat_team"],
                "argument_side":    "neutral"
            })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_paragraphs, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== DONE ===)")
    print(f"Total paragraphs generated: {len(all_paragraphs)}")
    print(f"Output file: {OUTPUT_FILE}")


if __name__ == "__main__":
    build_stat_paragraphs()