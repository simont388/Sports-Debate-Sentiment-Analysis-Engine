import re

# Stage 5 — Score evidence chunks and determine verdict lean
# Final score = 0.5*LSI + 0.3*source_reliability + 0.2*specificity

ALPHA = 0.5   #LSI similarity weight
BETA = 0.3   #Source reliability weight
GAMMA = 0.2   #Specificity weight

def compute_specificity(chunk, claim):
    text = chunk.get("text", "").lower()
    entities = [e.lower() for e in claim.get("entities", [])]
    stat_dims = [s.lower() for s in claim.get("stat_dimensions", [])]
    concepts = [c.lower() for c in claim.get("concepts", [])]

    score = 0.0
    checks = 3
    
    # Require at least one entity to be present in the text if entities are defined
    entity_found = False
    if not entities:
        # If no entities in claim, we can't really check for them, so we pass this check
        entity_found = True
    else:
        for entity in entities:
            # Check for full name or significant parts
            parts = entity.split()
            # If it's a multi-word name, check for the whole thing or the last name (usually unique enough)
            if entity in text or (len(parts) > 1 and parts[-1] in text):
                entity_found = True
                break
    
    if not entity_found:
        return 0.0  # Strict penalty: no entity match means 0 specificity

    score += 1.0
    
    stat_found = any(stat in text for stat in stat_dims if len(stat) > 2)

    stat_aliases = {
        "per": ["player efficiency", "per "],
        "ts%": ["true shooting", "ts%"],
        "bpm": ["box plus minus", "bpm "],
        "vorp": ["value over replacement", "vorp"],
        "ws": ["win shares"],
        "ws/48": ["win shares per 48 minutes", "ws/48"],
        "ppg": ["points per game", "ppg", "scoring average"],
        "rpg": ["rebounds per game", "rpg", "rebounding average"],
        "apg": ["assists per game", "apg", "assists average"],
    }

    for stat in stat_dims:
        aliases = stat_aliases.get(stat.lower(), [])
        if any(alias in text for alias in aliases):
            stat_found = True
            break
    
    if stat_found:
        score += 1.0

    concept_found = any(concept in text for concept in concepts if len(concept) > 3)
    if concept_found:
        score += 1.0
    
    return score / checks

def score_chunk(chunk, claim):
    lsi_sim = chunk.get("lsi_score", 0.0)
    source_weight = chunk.get("source_weight", 0.6)
    specificity = compute_specificity(chunk, claim)

    final_score = ALPHA*lsi_sim + BETA*source_weight + GAMMA*specificity

    chunk["specificity"] = round(specificity, 3)
    chunk["final_score"] = round(final_score, 3)

    return chunk

def deduplicate_within_set(chunks):
    seen = {}
    for chunk in chunks:
        key = chunk.get("chunk_id", chunk.get("text", "")[:100])
        if key not in seen:
            seen[key] = chunk
        else:
            if chunk.get("final_score", 0) > seen[key].get("final_score", 0):
                seen[key] = chunk

    return list(seen.values())

def deduplicate_across_sets(results_a, results_b):
    def get_key(chunk):
        return chunk.get("chunk_id", chunk.get("text", "")[:100])

    keys_a = {get_key(c): c for c in results_a}
    keys_b = {get_key(c): c for c in results_b}
    overlap_keys = set(keys_a.keys()) & set(keys_b.keys())

    final_a = []
    final_b = []

    for key, chunk in keys_a.items():
        if key not in overlap_keys:
            final_a.append(chunk)
        else:
            score_a = chunk.get("final_score", 0)
            score_b = keys_b[key].get("final_score", 0)
            if score_a >= score_b:
                final_a.append(chunk)

    for key, chunk in keys_b.items():
        if key not in overlap_keys:
            final_b.append(chunk)
        else:
            score_a = keys_a[key].get("final_score", 0)
            score_b = chunk.get("final_score", 0)
            if score_b > score_a:
                final_b.append(chunk)

    return final_a, final_b

def score_argument_sides(chunks):
    for chunk in chunks:
        retrieval_type = chunk.get("retrieval_type", "lsi")
        specificity = chunk.get("specificity", 0.0)
        if retrieval_type == "stat_lookup":
            chunk["side_confidence"] = "high"
        elif specificity >= 0.67:
            chunk["side_confidence"] = "medium"
        else:
            chunk["side_confidence"] = "low"
    return chunks

def score_results(claim):
    results_a = claim.get("results_a", [])
    results_b = claim.get("results_b", [])
    
    results_a = [score_chunk(chunk, claim) for chunk in results_a]
    results_b = [score_chunk(chunk, claim) for chunk in results_b]

    # Phase 1: Filter out LSI results with zero specificity (no entity match)
    results_a = [c for c in results_a if c.get("retrieval_type") == "stat_lookup" or c.get("specificity", 0) > 0]
    results_b = [c for c in results_b if c.get("retrieval_type") == "stat_lookup" or c.get("specificity", 0) > 0]

    results_a = deduplicate_within_set(results_a)
    results_b = deduplicate_within_set(results_b)
    results_a, results_b = deduplicate_across_sets(results_a, results_b)

    results_a = score_argument_sides(results_a)
    results_b = score_argument_sides(results_b)

    results_a.sort(key=lambda x: x["final_score"], reverse=True)
    results_b.sort(key=lambda x: x["final_score"], reverse=True)

    top_a_scores = [c["final_score"] for c in results_a[:5]]
    top_b_scores = [c["final_score"] for c in results_b[:5]]

    avg_a = sum(top_a_scores) / len(top_a_scores) if top_a_scores else 0
    avg_b = sum(top_b_scores) / len(top_b_scores) if top_b_scores else 0

    total = avg_a + avg_b
    if total > 0:
        confidence = avg_a / total
    else:
        confidence = 0.5
    
    if confidence > 0.53:
        verdict_lean = "supports the claim"
    elif confidence < 0.47:
        verdict_lean = "refutes the claim"
    else:
        verdict_lean = "mixed evidence"
    
    claim["scored_a"] = results_a
    claim["scored_b"] = results_b
    claim["confidence_score"] = confidence
    claim["verdict_lean"] = verdict_lean

    return claim

def test_scoring():
    from query_parser import parse_query, load_spacy
    from sentiment import SentimentAnalyzer
    from query_expansion import expand_query, load_lexicon, load_refutation_map
    from retrieval import retrieve, load_paragraph_model, load_stopwords

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
        claim = score_results(claim)
 
        print(f"\n  Verdict lean:      {claim['verdict_lean']}")
        print(f"  Confidence score:  {claim['confidence_score']}")
 
        print(f"\n  TOP 5 SUPPORTING (Set A):")
        for i, chunk in enumerate(claim["scored_a"][:5], 1):
            print(f"    {i}. score={chunk['final_score']:.3f} "
                  f"lsi={chunk['lsi_score']:.3f} "
                  f"spec={chunk['specificity']:.3f} "
                  f"conf={chunk['side_confidence']}")
            print(f"       [{chunk['source']}] {chunk['text'][:100]}...")
 
        print(f"\n  TOP 5 REFUTING (Set B):")
        for i, chunk in enumerate(claim["scored_b"][:5], 1):
            print(f"    {i}. score={chunk['final_score']:.3f} "
                  f"lsi={chunk['lsi_score']:.3f} "
                  f"spec={chunk['specificity']:.3f} "
                  f"conf={chunk['side_confidence']}")
            print(f"       [{chunk['source']}] {chunk['text'][:100]}...")

if __name__ == "__main__":
    test_scoring()