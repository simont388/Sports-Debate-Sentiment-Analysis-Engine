import os
import pickle
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from preprocessing.stats_template import (
    generate_trend_paragraph,
    generate_season_paragraph,
    generate_advanced_season_paragraph,
    generate_comparison_paragraph,
    safe_float,
    PLAYER_WINDOWS
)
from preprocessing.build_lsi import (
    tokenize_and_clean,
    compute_tfidf,
    query_projection,
)

# Stage 4 — Retrieve evidence via LSI semantic search and stat lookups

PARAGRAPH_MODEL_PATH = "models/lsi_paragraphs.pkl"
DOCUMENT_MODEL_PATH = "models/lsi_documents.pkl"
STATS_DIR = "data/stats"
STOPWORDS_FILE = "config/common_words"
TOP_K_PER_QUERY = 5
TOP_K_TOTAL = 20
STAT_SOURCE_WEIGHT = 1.0

stemmer = SnowballStemmer("english")

def load_stopwords(filepath=STOPWORDS_FILE):
    if not os.path.exists(filepath):
        return {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "of", "in", "on", "at", "to", "for", "with", "by", "from",
            "and", "or", "but", "not", "this", "that", "it", "he", "she",
            "they", "we", "you", "his", "her", "their", "its", "our", "my",
            "as", "if", "so", "up", "out", "about", "into", "then", "than",
            "also", "just", "more", "no", "all", "one", "who", "which"
        }
    with open(filepath) as f:
        return set(x.strip() for x in f.readlines())

def load_paragraph_model(path=PARAGRAPH_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
    
def load_document_model(path=DOCUMENT_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_relevant_debates(claim, doc_model, stopwords, top_n=5):
    # Use document-level LSI to find which debates are most relevant to the query
    Uk = doc_model["Uk"]
    Sk_inv = doc_model["Sk_inv"]
    Vtk = doc_model["Vtk"]
    word_map = doc_model["word_map"]
    doc_freqs = doc_model["doc_freqs"]
    N = doc_model["N"]
    doc_keys = doc_model["doc_keys"]

    all_queries = claim.get("query_set_a", []) + claim.get("query_set_b", [])
    if not all_queries:
        return set()

    best_scores = {}
    for query_str in all_queries:
        tokens = tokenize_and_clean(query_str, stopwords)
        if not tokens:
            continue
        tfidf_vec = compute_tfidf(tokens, doc_freqs, N)
        q_proj = query_projection(tfidf_vec, word_map, Uk, Sk_inv)
        dots = q_proj @ Vtk
        q_norm = np.linalg.norm(q_proj)
        doc_norms = np.linalg.norm(Vtk, axis=0)
        scores = dots / (doc_norms * q_norm + 1e-10)
        top_indeces = np.argsort(-scores)[:top_n]
        for idx in top_indeces:
            score = float(scores[idx])
            if idx not in best_scores or score > best_scores[idx]:
                best_scores[idx] = score

    relevant_debates = set()
    for idx in best_scores:
        key = doc_keys[idx]
        debate = key.split("::")[0]
        relevant_debates.add(debate)
    return relevant_debates

def filter_paragraph_indices(paragraphs, relevant_debates):
    # Narrow paragraph search to only those tagged with relevant debates
    indices = set()
    for i, p in enumerate(paragraphs):
        p_debates = p.get("debates", [])
        if any(d in relevant_debates for d in p_debates):
            indices.add(i)
    return indices

def lsi_search_query_set(query_strings, model, stopwords, argument_side, allowed_indices=None):
    Uk = model["Uk"]
    Sk_inv = model["Sk_inv"]
    Vtk = model["Vtk"]
    word_map = model["word_map"]
    doc_freqs = model["doc_freqs"]
    N = model["N"]
    paragraphs = model["paragraphs"]

    best_scores = {}

    for query_str in query_strings:
        tokens = tokenize_and_clean(query_str, stopwords)
        if not tokens:
            continue
        
        tfidf_vec = compute_tfidf(tokens, doc_freqs, N)
        q_proj = query_projection(tfidf_vec, word_map, Uk, Sk_inv)

        dots = q_proj @ Vtk
        q_norm = np.linalg.norm(q_proj)
        doc_norms = np.linalg.norm(Vtk, axis=0)
        scores = dots / (doc_norms * q_norm + 1e-10)

        if allowed_indices is not None:
            for i in range(len(scores)):
                if i not in allowed_indices:
                    scores[i] = -1.0

        top_indeces = np.argsort(-scores)[:TOP_K_PER_QUERY]
        for idx in top_indeces:
            chunk_id = paragraphs[idx].get("chunk_id", idx)
            score = float(scores[idx])
            if chunk_id not in best_scores or score > best_scores[chunk_id]:
                best_scores[chunk_id] = score

    chunk_lookup = {p.get("chunk_id", i): p for i,p in enumerate(paragraphs)}
    results = []
    for chunk_id, score in best_scores.items():
        chunk = chunk_lookup.get(chunk_id, [])
        result = dict(chunk)
        result["lsi_score"] = score
        result["argument_side"] = argument_side
        result["retrieval_type"] = "lsi"
        results.append(result)

    results.sort(key=lambda x: x["lsi_score"], reverse=True)
    return results[:TOP_K_TOTAL]

def run_lsi_retrieval(claim, model, stopwords, doc_model=None):
    query_set_a = claim.get("query_set_a", [])
    query_set_b = claim.get("query_set_b", [])

    allowed_indices = None
    if doc_model is not None:
        relevant_debates = get_relevant_debates(claim, doc_model, stopwords)
        if relevant_debates:
            allowed_indices = filter_paragraph_indices(model["paragraphs"], relevant_debates)

    results_a = lsi_search_query_set(query_set_a, model, stopwords, "supporting", allowed_indices)
    results_b = lsi_search_query_set(query_set_b, model, stopwords, "opposing", allowed_indices)
    return results_a, results_b

def get_player_csv_key(entity_name):
    # Map full player name to CSV filename key
    name_to_key = {
        "LeBron James": "lebron_james",
        "Stephen Curry": "stephen_curry",
        "Giannis Antetokounmpo": "giannis",
        "Kobe Bryant": "kobe_bryant",
        "Kawhi Leonard": "kawhi_leonard",
        "Russell Westbrook": "russell_westbrook",
        "Carmelo Anthony": "carmelo_anthony",
        "Draymond Green": "draymond_green",
        "Nikola Jokic": "nikola_jokic",
    }
    return name_to_key.get(entity_name)

def load_player_csv(player_key):
    pg_path = os.path.join(STATS_DIR, f"{player_key}_per_game.csv")
    adv_path = os.path.join(STATS_DIR, f"{player_key}_advanced.csv")

    df_pg = pd.read_csv(pg_path) if os.path.exists(pg_path) else None
    df_adv = pd.read_csv(adv_path) if os.path.exists(adv_path) else None

    return df_pg, df_adv

def stat_chunk(text, player, argument_side, stat=None):
    # Build a stat evidence chunk with max LSI score (objective data)
    return {
        "text": text,
        "source": "Basketball Reference",
        "source_weight": STAT_SOURCE_WEIGHT,
        "player": player,
        "season": "career" if stat else "various",
        "paragraph_type": "stat_lookup",
        "lsi_score": 1.0,  # stat lookups get max LSI score
        "argument_side": argument_side,
        "retrieval_type": "stat_lookup",
        "stat": stat or ""
    }

def run_trend_lookup(claim):
    # Trend claims: compare prime vs. recent season stats to detect decline or improvement
    entities = claim.get("entities", [])
    if not entities:
        return [], []
    
    entity = entities[0]
    player_key = get_player_csv_key(entity)
    if not player_key or player_key not in PLAYER_WINDOWS:
        return [], []
    
    config = PLAYER_WINDOWS[player_key]
    prime_seasons = config["prime"]
    recent_seasons = config["recent"]
    relevant_seasons = config.get("relevant_seasons", [])
    display = config["display"]
    df_pg, df_adv = load_player_csv(player_key)

    supporting = []
    refuting = []

    if relevant_seasons:
        if df_pg is not None:
            for _, row in df_pg[df_pg["Season"].isin(relevant_seasons)].iterrows():
                para = generate_season_paragraph(display, row)
                if para:
                    chunk = stat_chunk(para, display, "neutral")
                    chunk["season"] = row.get("Season", "")
                    supporting.append(chunk)
        if df_adv is not None:
            for _, row in df_adv[df_adv["Season"].isin(relevant_seasons)].iterrows():
                para = generate_advanced_season_paragraph(display, row)
                if para:
                    chunk = stat_chunk(para, display, "neutral")
                    chunk["season"] = row.get("Season", "")
                    supporting.append(chunk)
        return supporting, refuting

    pg_stats = ["PPG", "RPG", "APG", "FG%", "3P%"]
    if df_pg is not None:
        for stat in pg_stats:
            if stat not in df_pg.columns:
                continue
            para = generate_trend_paragraph(display, df_pg, stat, prime_seasons, recent_seasons)
            if not para:
                continue

            prime_vals = [safe_float(v) for v in df_pg[df_pg["Season"].isin(prime_seasons)][stat]
                          if safe_float(v) is not None]
            recent_vals = [safe_float(v) for v in df_pg[df_pg["Season"].isin(recent_seasons)][stat]
                           if safe_float(v) is not None]
            
            if prime_vals and recent_vals:
                prime_avg = sum(prime_vals) / len(prime_vals)
                recent_avg = sum(recent_vals) / len(recent_vals)
                # For trend claims, decline supports the negative narrative
                side = "supporting" if recent_avg < prime_avg else "opposing"
            else:
                side = "neutral"
            
            chunk = stat_chunk(para, display, side, stat)
            if side == "supporting":
                supporting.append(chunk)
            else:
                refuting.append(chunk)
    
    adv_stats = ["PER", "TS%", "BPM", "VORP", "WS", "WS/48"]
    if df_adv is not None:
        for stat in adv_stats:
            if stat not in df_adv.columns:
                continue
            para = generate_trend_paragraph(display, df_adv, stat, prime_seasons, recent_seasons)
            if not para:
                continue

            prime_vals = [safe_float(v) for v in df_adv[df_adv["Season"].isin(prime_seasons)][stat]
                          if safe_float(v) is not None]
            recent_vals = [safe_float(v) for v in df_adv[df_adv["Season"].isin(recent_seasons)][stat]
                           if safe_float(v) is not None]
            
            if prime_vals and recent_vals:
                prime_avg = sum(prime_vals) / len(prime_vals)
                recent_avg = sum(recent_vals) / len(recent_vals)
                side = "supporting" if recent_avg < prime_avg else "opposing"
            else:
                side = "neutral"
            
            chunk = stat_chunk(para, display, side, stat)
            if side == "supporting":
                supporting.append(chunk)
            else:
                refuting.append(chunk)
    
    return supporting, refuting

def run_comparison_lookup(claim):
    # Comparison claims: head-to-head stat comparison between two players
    entities = claim.get("entities", [])
    stat_dims = claim.get("stat_dimensions", [])
    if len(entities) < 2:
        return [], []
    
    entity_a = entities[0]
    entity_b = entities[1]
    key_a = get_player_csv_key(entity_a)
    key_b = get_player_csv_key(entity_b)
    if not key_a or not key_b:
        return [], []
    
    df_a, df_adv_a = load_player_csv(key_a)
    df_b, df_adv_b = load_player_csv(key_b)
    if df_a is None or df_b is None:
        return [], []
    
    seasons_a = set(df_a["Season"].tolist())
    seasons_b = set(df_b["Season"].tolist())
    overlap = sorted(seasons_a.intersection(seasons_b))

    supporting = []
    refuting = []
    pg_stats = [s for s in stat_dims if s in ["PPG", "FG%", "3P%", "FT%"]]
    if not pg_stats:
        pg_stats = ["PPG", "FG%"]
    
    for season in overlap:
        for stat in pg_stats:
            if stat not in df_a.columns or stat not in df_b.columns:
                continue
            para = generate_comparison_paragraph(entity_a, entity_b, df_a, df_b, stat, season)
            if not para:
                continue

            row_a = df_a[df_a["Season"] == season]
            row_b = df_b[df_b["Season"] == season]
            val_a = safe_float(row_a.iloc[0].get(stat)) if not row_a.empty else None
            val_b = safe_float(row_b.iloc[0].get(stat)) if not row_b.empty else None

            if val_a is not None and val_b is not None:
                # Player A outperforming B supports the claim "A is better"
                side = "supporting" if val_a >= val_b else "opposing"
            else:
                side = "neutral"
            
            chunk = stat_chunk(para, f"{entity_a} vs {entity_b}", side, stat)
            if side == "supporting":
                supporting.append(chunk)
            else:
                refuting.append(chunk)

    if df_adv_a is not None and df_adv_b is not None:
        adv_stats = ["PER", "TS%", "BPM", "WS", "WS/48"]
        for stat in adv_stats:
            if stat not in df_adv_a.columns or stat not in df_adv_b.columns:
                continue
            para = generate_comparison_paragraph(entity_a, entity_b, df_adv_a, df_adv_b, stat, season=None)
            if not para:
                continue

            chunk = stat_chunk(para, f"{entity_a} vs {entity_b}", "neutral", stat)
            supporting.append(chunk)
    
    return supporting, refuting

def run_causal_lookup(claim):
    # Causal claims (e.g. "can't win without co-star", "ruined the team")
    # can't be evaluated from individual season stats alone. Individual PPG,
    # rebounds, etc. don't prove or disprove claims about team dynamics,
    # championship dependence, or player impact on team success.
    # Let LSI retrieval find the relevant evidence from articles and Wikipedia.
    return [], []

def retrieve(claim, model=None, stopwords=None, doc_model=None):
    if stopwords is None:
        stopwords = load_stopwords()
    if model is None:
        print(f"Loading LSI model...")
        model = load_paragraph_model()
    
    retrieval_strategy = claim.get("retrieval_strategy", ["lsi_retrieval"])

    lsi_a, lsi_b = run_lsi_retrieval(claim, model, stopwords, doc_model)
    stat_a, stat_b = [], []
    
    if "year_over_year_comparison" in retrieval_strategy:
        stat_a, stat_b = run_trend_lookup(claim)
    elif "side_by_side_stat_lookup" in retrieval_strategy:
        stat_a, stat_b = run_comparison_lookup(claim)
    elif "conditional_stat_lookup" in retrieval_strategy:
        stat_a, stat_b = run_causal_lookup(claim)
    
    # Merge stat lookups (higher priority) before LSI results
    results_a = stat_a + lsi_a
    results_b = stat_b + lsi_b

    claim["results_a"] = results_a
    claim["results_b"] = results_b
    return claim

def test_retrieval():
    from query_parser import parse_query, load_spacy
    from sentiment import SentimentAnalyzer
    from query_expansion import expand_query, load_lexicon, load_refutation_map

    stopwords = load_stopwords()
    model = load_paragraph_model()
    lexicon = load_lexicon()
    refutation_map = load_refutation_map()
    nlp = load_spacy()
    analyzer = SentimentAnalyzer()

    test_queries = [
        "LeBron is washed",
        "Kobe was a better scorer than LeBron",
        "Kawhi Leonard is overrated",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        claim = parse_query(query, nlp=nlp, lexicon=lexicon)
        claim = analyzer.enrich_claim(claim)
        claim = expand_query(claim, lexicon=lexicon, refutation_map=refutation_map)
        claim = retrieve(claim, model=model, stopwords=stopwords)

        print(f"\n  TOP 5 SUPPORTING (Set A):")
        for i, chunk in enumerate(claim["results_a"][:5], 1):
            print(f"    {i}. [{chunk['retrieval_type'].upper()}] "
                  f"[{chunk['source']}] "
                  f"score={chunk['lsi_score']:.3f}")
            print(f"       {chunk['text'][:120]}...")
 
        print(f"\n  TOP 5 REFUTING (Set B):")
        for i, chunk in enumerate(claim["results_b"][:5], 1):
            print(f"    {i}. [{chunk['retrieval_type'].upper()}] "
                  f"[{chunk['source']}] "
                  f"score={chunk['lsi_score']:.3f}")
            print(f"       {chunk['text'][:120]}...")

if __name__ == "__main__":
    test_retrieval()