import json
import os

# Stage 3 — Query expansion into supporting (Set A) and refuting (Set B) query sets

LEXICON_PATH = "config/word_mapping.json"
REFUTATION_MAP = "config/refutation.json"
MAX_QUERIES_PER_SET = 5
# Filter out non-sports concepts that GloVe/WordNet might expand to
EXPANSION_STOPLIST = {
    "actress", "actor", "film", "movie", "television", "tv", "cinema",
    "theater", "theatre", "director", "producer", "cast", "role",
    "financial", "economic", "stock", "market", "quarter", "fiscal"
}

def load_lexicon(path=LEXICON_PATH):
    with open(path, "r") as f:
        return json.load(f)
    
def load_refutation_map(path=REFUTATION_MAP):
    with open(path, "r") as f:
        return json.load(f)
    
def filter_concepts(concepts):
    return [c for c in concepts if c.lower() not in EXPANSION_STOPLIST]

def build_query_string(entity, concepts, stat_dimensions=None):
    # Join entity + concepts + short stat names into a single query string
    parts = [entity] if entity and entity != "various" else []

    clean_concepts = filter_concepts(concepts)
    for concept in clean_concepts:
        if concept and concept not in parts:
            parts.append(concept)
    
    if stat_dimensions:
        for stat in stat_dimensions:
            if len(stat.split()) <= 2 and stat not in parts:
                parts.append(stat)
    
    return " ".join(parts)

def build_comparison_query(entity_a, entity_b, concepts):
    # Build a query string for two-player comparison claims
    parts = []
    if entity_a:
        parts.append(entity_a)
    if entity_b:
        parts.append(entity_b)
    clean_concepts = filter_concepts(concepts)  
    for concept in clean_concepts:
        if concept and concept not in parts:
            parts.append(concept)
    return " ".join(parts)

def build_supporting_queries(claim, lexicon):
    # Set A — queries that support the original claim
    entities = claim.get("entities", [])
    expanded = claim.get("expanded_concepts", [])
    stat_dimensions = claim.get("stat_dimensions", [])
    slang_terms = claim.get("slang_terms", [])
    claim_type = claim.get("claim_type", [])

    primary_entity = entities[0] if entities else ""
    secondary_entity = entities[1] if len(entities) > 1 else ""

    queries = []

    if expanded:
        q = build_query_string(primary_entity, expanded[:5], stat_dimensions[:2])
        queries.append(q)
    
    if stat_dimensions:
        q = build_query_string(primary_entity, stat_dimensions[:4])
        queries.append(q)

    # Build queries from slang-derived concepts (e.g. "washed" -> "decline", "aging")
    slang_concepts = []
    for term in slang_terms:
        for concept in lexicon["terms"].get(term, {}).get("concepts", []):
            if concept not in slang_concepts:
                slang_concepts.append(concept)
    if slang_concepts:
        q = build_query_string(primary_entity, slang_concepts[:4])
        queries.append(q)
    
    if claim_type == "direct_comparison" and secondary_entity:
        q = build_comparison_query(primary_entity, secondary_entity, slang_concepts[:3])
        queries.append(q)
    
    if expanded and stat_dimensions:
        mixed = expanded[:2] + [stat_dimensions[0]]
        q = build_query_string(primary_entity, mixed)
        queries.append(q)
    
    seen = set()
    unique = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)
    
    return unique[:MAX_QUERIES_PER_SET]

def build_refuting_queries(claim, refutation_map):
    # Set B — queries that refute the claim using counter-argument concepts
    entities          = claim.get("entities", [])
    claim_type        = claim.get("claim_type", "trend")
    sentiment         = claim.get("sentiment_direction", "negative")
    stat_dimensions   = claim.get("stat_dimensions", [])
    primary_entity   = entities[0] if entities else ""
    secondary_entity = entities[1] if len(entities) > 1 else ""
    
    map_key = f"{claim_type}_{sentiment}"
    refutation_entry = refutation_map.get(map_key, {})
    refutation_concepts = refutation_entry.get("concepts", [])

    if claim_type == "direct_comparison" and secondary_entity:
        refuting_primary = secondary_entity
        refuting_secondary = primary_entity
    else:
        refuting_primary = primary_entity
        refuting_secondary = secondary_entity

    if not refutation_concepts:
        refutation_concepts = ["elite performance", "high level", "successful", "proven"]
    
    queries = []

    if refutation_concepts:
        q = build_query_string(refuting_primary, refutation_concepts[:5])
        queries.append(q)
    
    if refutation_concepts and stat_dimensions:
        q = build_query_string(refuting_primary, refutation_concepts[:3], stat_dimensions[:2])
        queries.append(q)
    
    if len(refutation_concepts) >= 2:
        q = build_query_string(refuting_primary, refutation_concepts[:2])
        queries.append(q)
    
    if claim_type == "direct_comparison" and secondary_entity:
        q = build_comparison_query(refuting_primary, refuting_secondary, refutation_concepts[:3])
        queries.append(q)
    
    if len(refutation_concepts) >= 4:
        q = build_query_string(refuting_primary, refutation_concepts[2:5])
        queries.append(q)

    seen = set()
    unique = []
    for q in queries:
        if q and q not in seen:
            seen.add(q)
            unique.append(q)

    return unique[:MAX_QUERIES_PER_SET]

def expand_query(claim, lexicon=None, refutation_map=None):
    if lexicon is None:
        lexicon = load_lexicon()
    if refutation_map is None:
        refutation_map = load_refutation_map()

    query_set_a = build_supporting_queries(claim, lexicon)
    query_set_b = build_refuting_queries(claim, refutation_map)

    claim["query_set_a"] = query_set_a
    claim["query_set_b"] = query_set_b

    return claim

def test_query_expansion():
    from query_parser import parse_query, load_spacy
    from sentiment import SentimentAnalyzer

    lexicon = load_lexicon()
    refutation_map = load_refutation_map()
    nlp = load_spacy()
    analyzer = SentimentAnalyzer()

    test_queries = [
        "LeBron is washed",
        "Steph Curry is a system player",
        "Giannis can't win a championship without a co-star",
        "Kobe was a better scorer than LeBron",
        "The 2017 Warriors are the greatest team of all time",
        "Kawhi Leonard is overrated",
        "Westbrook ruined the Lakers",
        "Carmelo Anthony was never a winner because he was selfish",
        "Draymond Green is the most important player on the Warriors",
        "Nikola Jokic is the most complete player of all time",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        claim = parse_query(query, nlp=nlp, lexicon=lexicon)
        claim = analyzer.enrich_claim(claim)
        claim = expand_query(claim, lexicon=lexicon, refutation_map=refutation_map)

        print(f"\n  SET A (supporting):")
        for i, q in enumerate(claim["query_set_a"], 1):
            print(f"    {i}. {q}")
 
        print(f"\n  SET B (refuting):")
        for i, q in enumerate(claim["query_set_b"], 1):
            print(f"    {i}. {q}")

if __name__ == "__main__":
    test_query_expansion()