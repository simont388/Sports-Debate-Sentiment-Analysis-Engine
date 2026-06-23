import json
import re
import spacy

# Stage 1 — Parse the raw query into a structured claim dict

def load_lexicon(path="config/word_mapping.json"):
    with open(path, "r") as f:
        return json.load(f)
    
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )

def extract_entities(text, nlp, player_aliases):
    # Use spaCy NER to find person names, then resolve to canonical names via aliases
    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip().lower()
            first_word = name.split()[0]
            # Try full name first, then first name, fall back to original text
            resolved = player_aliases.get(name) or player_aliases.get(first_word) or ent.text.strip()
            if resolved not in entities:
                entities.append(resolved)
    
    # Also check for nicknames/aliases not caught by NER (e.g. "giannis", "greek freak")
    text_lower = text.lower()
    for alias, full_name in player_aliases.items():
        if alias in text_lower and full_name not in entities:
            entities.append(full_name)
    
    return entities

def detect_negation(text, negation_words, slang_terms):
    # Check if a negation word appears near a slang term (window of +/- 3 tokens)
    tokens = text.lower().split()
    for i, token in enumerate(tokens):
        if token in negation_words:
            window_start = max(0, i-3)
            window_end = min(len(tokens), i+4)
            window = " ".join(tokens[window_start:window_end])
            for term in slang_terms:
                if term in window:
                    return True
    return False

def detect_causal_clause(text, causal_tokens):
    # Look for causal keywords like "because", "ruined", "needs"
    text_lower = text.lower()
    for token in causal_tokens:
        if token in text_lower:
            return token
    return None

def detect_temporal_signal(text, temporal_signals):
    # Detect time-related words like "decline", "peak", "used to"
    text_lower = text.lower()
    for signal in temporal_signals:
        if signal in text_lower:
            return signal
    return None

def extract_stat_dimension(text, stat_dimension_aliases):
    # Map stat mentions in text to canonical keys (e.g. "scoring" -> "PPG")
    text_lower = text.lower()
    for alias, stat_key in stat_dimension_aliases.items():
        if alias in text_lower:
            return stat_key
    return None

def detect_slang_terms(text, terms):
    # Match slang terms from lexicon, sorted longest-first to avoid partial matches
    text_lower = text.lower()
    matched = []
    sorted_terms = sorted(terms.keys(), key=len, reverse=True)
    for term in sorted_terms:
        if term in text_lower:
            matched.append(term)
    return matched

def classify_claim_type(entities, slang_terms, causal_token, temporal_signal, 
                        stat_dimension, terms, claim_type_rules):
    if len(entities) >= 2:
        return claim_type_rules.get("two_entities_override", "direct_comparison")
    
    if causal_token:
        return "causal"
    
    # Infer claim type from the slang term's configured type
    for term in slang_terms:
        if term in terms:
            term_claim_type = terms[term].get("claim_type")
            if term_claim_type:
                return term_claim_type

    if temporal_signal:
        return "trend"
    
    if stat_dimension and len(entities) == 1:
        return "trend"
    
    return claim_type_rules.get("default_single_entity", "trend")

def parse_query(query, nlp=None, lexicon=None):
    # Main entry point: returns a structured claim dict with all detected signals
    if lexicon is None:
        lexicon = load_lexicon()
    if nlp is None:
        nlp = load_spacy()

    terms = lexicon["terms"]
    negation_words = lexicon["negation_words"]
    causal_tokens = lexicon["causal_tokens"]
    player_aliases = lexicon["player_aliases"]
    stat_dimension_aliases = lexicon["stat_dimension_aliases"]
    claim_type_rules = lexicon["claim_type_rules"]
    temporal_signals = claim_type_rules["temporal_signals"]["tokens"]

    entities = extract_entities(query, nlp, player_aliases)
    causal_token = detect_causal_clause(query, causal_tokens)
    temporal_signal = detect_temporal_signal(query, temporal_signals)
    stat_dimension = extract_stat_dimension(query, stat_dimension_aliases)
    slang_terms = detect_slang_terms(query, terms)
    negation = detect_negation(query, negation_words, slang_terms)

    claim_type = classify_claim_type(entities, slang_terms, causal_token, temporal_signal, 
                                     stat_dimension, terms, claim_type_rules)
    
    # Collect stat dimensions from slang term definitions
    stat_dimensions = []
    if stat_dimension:
        stat_dimensions = [stat_dimension]
    else:
        for term in slang_terms:
            if term in terms:
                for s in terms[term].get("stat_dimensions", []):
                    if s not in stat_dimensions:
                        stat_dimensions.append(s)

        concepts = []
        for term in slang_terms:
            if term in terms:
                for c in terms[term].get("concepts", []):
                    if c not in concepts:
                        concepts.append(c)

        # Determine sentiment direction from the first slang term, flip if negation detected
        sentiment_direction = "negative"
        if slang_terms:
            first_term = slang_terms[0]
            if first_term in terms:
                sentiment_direction = terms[first_term].get("sentiment_direction", "negative")
        if negation:
            if sentiment_direction == "negative":
                sentiment_direction = "positive"
            elif sentiment_direction == "positive":
                sentiment_direction = "negative"

        related_debates = []
        for term in slang_terms:
            if term in terms:
                for d in terms[term].get("related_debates", []):
                    if d not in related_debates:
                        related_debates.append(d)

        # Select retrieval strategy based on claim type
        retrieval_strategy = []
        if claim_type == "trend":
            retrieval_strategy = ["year_over_year_comparison", "lsi_retrieval"]
        elif claim_type == "direct_comparison":
            retrieval_strategy = ["side_by_side_stat_lookup", "lsi_retrieval"]
        elif claim_type == "causal":
            retrieval_strategy = ["conditional_stat_lookup", "lsi_retrieval"]
        elif claim_type in ["legitimacy", "historical_comparison"]:
            retrieval_strategy = ["lsi_retrieval"]

        claim = {
            "raw_query":          query,
            "entities":           entities,
            "slang_terms":        slang_terms,
            "concepts":           concepts,
            "stat_dimensions":    stat_dimensions,
            "claim_type":         claim_type,
            "sentiment_direction": sentiment_direction,
            "negation":           negation,
            "causal_token":       causal_token,
            "temporal_signal":    temporal_signal,
            "related_debates":    related_debates,
            "retrieval_strategy": retrieval_strategy
        }

        return claim
    
def test_parser():
    lexicon = load_lexicon()
    nlp = load_spacy()

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
        for key, value in claim.items():
            if key != "raw_query":
                print(f"  {key:<22}: {value}")
 
 
if __name__ == "__main__":
    test_parser()