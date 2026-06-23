import sys
import time
import os
from query_parser import parse_query, load_spacy
from sentiment import SentimentAnalyzer
from query_expansion import expand_query, load_lexicon, load_refutation_map
from retrieval import retrieve, load_paragraph_model, load_document_model, load_stopwords
from scoring import score_results
from verdict import generate_verdict

# Entry point — orchestrates the 6-stage debate retrieval pipeline
# Usage: python main.py [query | --all | --interactive]

FIXED_DEBATES = [
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
OUTPUT_FILE = "results/all_debate_results.txt"
os.makedirs("results", exist_ok=True)

def run_pipeline(query, resources):
    nlp = resources["nlp"]
    analyzer = resources["analyzer"]
    lexicon = resources["lexicon"]
    refutation_map = resources["refutation_map"]
    model = resources["model"]
    doc_model = resources["doc_model"]
    stopwords = resources["stopwords"]

    print(f"\nRunning pipeline for: '{query}'")
    print(f"{'.'*50}")
 
    # Stage 1 — Parse
    print(f"[1/6] Parsing query...")
    claim = parse_query(query, nlp=nlp, lexicon=lexicon)
    print(f"Claim type: {claim['claim_type']} | "
          f"Entities: {claim['entities']} | "
          f"Slang: {claim['slang_terms']}")
    
    # Stage 2 — Sentiment enrichment
    print(f"[2/6] Enriching with GloVe/WordNet...")
    claim = analyzer.enrich_claim(claim)
    print(f"Expanded concepts: {len(claim.get('expanded_concepts', []))} terms")
 
    # Stage 3 — Query expansion
    print(f"[3/6] Building query sets...")
    claim = expand_query(claim, lexicon=lexicon, refutation_map=refutation_map)
    print(f"Set A: {len(claim['query_set_a'])} queries | "
          f"Set B: {len(claim['query_set_b'])} queries")
 
    # Stage 4 — Retrieval
    print(f"[4/6] Retrieving evidence...")
    claim = retrieve(claim, model=model, stopwords=stopwords, doc_model=doc_model)
    print(f"Results A: {len(claim['results_a'])} chunks | "
          f"Results B: {len(claim['results_b'])} chunks")
 
    # Stage 5 — Scoring
    print(f"[5/6] Scoring and deduplicating...")
    claim = score_results(claim)
    print(f"Verdict lean: {claim['verdict_lean']} | "
          f"Confidence: {claim['confidence_score']:.3f}")
 
    # Stage 6 — Verdict
    print(f"[6/6] Generating verdict...")
    claim = generate_verdict(claim)
    print(f"Method: {claim['verdict']['method']}")

    return claim

def load_resources():
    print("Loading resources...")
    print("[1/6] Loading spaCy...")
    nlp = load_spacy()

    print("[2/6] Loading GloVe vectors (this takes a moment)...")
    analyzer = SentimentAnalyzer()
  
    print("[3/6] Loading lexicon and refutation map...")
    lexicon = load_lexicon()
    refutation_map = load_refutation_map()
  
    print("[4/6] Loading stopwords...")
    stopwords = load_stopwords()
 
    print("[5/6] Loading LSI paragraph model...")
    model = load_paragraph_model()

    print("[6/6] Loading LSI document model...")
    doc_model = load_document_model()
  
    print("All resources loaded.\n")

    return {
        "nlp": nlp,
        "analyzer": analyzer,
        "lexicon": lexicon,
        "refutation_map": refutation_map,
        "stopwords": stopwords,
        "model": model,
        "doc_model": doc_model,
    }

def run_single(query, resources):
    claim = run_pipeline(query, resources)
    print_verdict(claim)

def run_all(resources):
    reset_output_file()
    print(f"\nRunning all {len(FIXED_DEBATES)} fixed debates...\n")
    for i, query in enumerate(FIXED_DEBATES, 1):
        print(f"[{i}/{len(FIXED_DEBATES)}]", end="")
        claim = run_pipeline(query, resources)
        print_verdict(claim, to_file=True)
        # Small delay between queries to avoid Ollama timeout
        if i < len(FIXED_DEBATES):
            time.sleep(5)

def run_interactive(resources):
    print("\n" + "="*65)
    print("SPORTS DEBATE RATIONALE RETRIEVAL SYSTEM")
    print("Type a sports take to get evidence for and against it.")
    print("Commands: 'quit' to exit | 'all' to run fixed debates | ")
    print("'list' to see example debates")
    print("="*65)

    while True:
        print()
        query = input("Enter a sports take: ").strip()
 
        if not query:
            continue
 
        if query.lower() in ["quit", "exit", "q"]:
            print("\nExiting. Goodbye.")
            break
 
        if query.lower() == "all":
            run_all(resources)
            continue
 
        if query.lower() == "list":
            print("\nFixed debates:")
            for i, debate in enumerate(FIXED_DEBATES, 1):
                print(f"{i}. {debate}")
            continue
 
        try:
            claim = run_pipeline(query, resources)
            print_verdict(claim)
        except Exception as e:
            print(f"\nError processing query: {e}")
            print("Please try a different query.")
            continue

def print_verdict(claim, to_file=False):
    verdict = claim.get("verdict", {})
    verdict_text = verdict.get("verdict_text", "No verdict generated.")
    verdict_lean = verdict.get("verdict_lean", "unknown").upper()
    confidence = verdict.get("confidence", 0.5)
    method = verdict.get("method", "unknown")
    top_a = verdict.get("top_supporting", [])
    top_b = verdict.get("top_refuting", [])

    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"QUERY: {claim['raw_query']}")
    lines.append(f"{'='*65}")
    lines.append(f"\nVERDICT: {verdict_lean}")
    lines.append(f"Confidence: {confidence:.3f}  |  Method: {method}")
    lines.append(f"\n{verdict_text}")
    lines.append(f"\n--- SUPPORTING EVIDENCE ---")
    if top_a:
        for i, chunk in enumerate(top_a, 1):
            source = chunk.get("source", "Unknown")
            score = chunk.get("final_score", 0)
            confidence_flag = chunk.get("side_confidence", "low")
            text = chunk.get("text", "")[:200]
            lines.append(f"\n{i}. [{source}] score={score:.3f} [{confidence_flag}]")
            lines.append(f"{text}...")
    else:
        lines.append("No supporting evidence found")

    lines.append(f"\n--- REFUTING EVIDENCE ---")
    if top_b:
        for i, chunk in enumerate(top_b, 1):
            source = chunk.get("source", "Unknown")
            score = chunk.get("final_score", 0)
            confidence_flag = chunk.get("side_confidence", "low")
            text = chunk.get("text", "")[:200]
            lines.append(f"\n{i}. [{source}] score={score:.3f} [{confidence_flag}]")
            lines.append(f"{text}...")
    else:
        lines.append("No refuting evidence found.")

    lines.append(f"\n{'='*65}\n")
    output = "\n".join(lines)

    if to_file:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Result written to {OUTPUT_FILE}\n")
    else:
        print(output)

def reset_output_file():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"SPORTS DEBATE RESULTS\n")
        f.write(f"{'='*65}\n")

if __name__ == "__main__":
    resources = load_resources()

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--all":
            run_all(resources)
        elif arg == "--interactive":
            run_interactive(resources)
        else:
            # Treat the argument as a query
            query = " ".join(sys.argv[1:])
            run_single(query, resources)
    
    else:
        # Default to interactive
        run_interactive(resources)